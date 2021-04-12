from math import prod
from pickle import NONE
from rlflows.models import NormalizingFlowsPolicy
from typing import List
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch.distributions import Categorical, Normal, Independent, TransformedDistribution


class TorchGaussianMixtureDistribution(TorchDistributionWrapper):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return prod((2, model_config['custom_model_config']['num_gaussians']) + action_space.shape)

    def __init__(self, inputs: List[torch.Tensor], model: TorchModelV2):
        super(TorchDistributionWrapper, self).__init__(inputs, model)
        assert len(inputs.shape) == 2
        self.batch_size = self.inputs.shape[0]
        self.num_gaussians = model.model_config['custom_model_config']['num_gaussians']
        self.monte_samples = model.model_config['custom_model_config']['monte_samples']
        inputs = torch.reshape(self.inputs, (self.batch_size, 2, self.num_gaussians, -1))
        self.action_dim = inputs.shape[-1]
        assert not torch.isnan(inputs).any(), "Input nan aborting"
        self.means = inputs[:, 0, :, :] # batch_size x num_gaussians x action_dim
        self.sigmas = torch.exp(inputs[:, 1, :, :]) # batch_size x num_gaussians x action_dim
        
        self.cat = Categorical(torch.ones(self.batch_size, self.num_gaussians, device=inputs.device, requires_grad=False))
        self.normals = Independent(Normal(self.means, self.sigmas), 1)

    def logp(self, actions: torch.Tensor):
        actions = actions.view(self.batch_size, 1, -1) # batch_size x 1 (broadcast to num gaussians) x action_dim
        mix_lps = self.cat.logits # batch_size x num_gaussians x action_dim
        normal_lps = self.normals.log_prob(actions) # batch_size x num_gaussians x action_dim
        assert not torch.isnan(mix_lps).any(), "output nan aborting"
        assert not torch.isnan(normal_lps).any(), "output nan aborting"
        return torch.logsumexp(mix_lps + normal_lps, dim=1) # reduce along num gaussians

    def deterministic_sample(self) -> torch.Tensor:
        self.last_sample = self.means[:, 0, :] # select the mode of the first gaussian
        return self.last_sample

    def __rsamples(self):
        """ Compute samples that can be differentiated through
        """
        # Using reparameterization trick i.e. rsample
        normal_samples = self.normals.rsample((self.monte_samples,)) # monte_samples x batch_size x num_gaussians x action_dim
        cat_samples = self.cat.sample((self.monte_samples,)) # monte_samples x batch_size
        # First we need to expand cat so that it has the same dimension as normal samples
        cat_samples = cat_samples.reshape(self.monte_samples, -1, 1, 1).expand(-1, -1, -1, self.action_dim)
        # We select the normal distribution based on the outputs of 
        # the categorical distribution
        return torch.gather(normal_samples, 2, cat_samples).squeeze(dim=2) # monte_samples x batch_size x action_dim


    def kl(self, q: ActionDistribution) -> torch.Tensor:
        """ KL(self || q) estimated with monte carlo sampling
        """
        rsamples = self.__rsamples().unbind(0)
        log_ratios = torch.stack([q.logp(rsample) - self.logp(rsample) for rsample in rsamples])
        assert not torch.isnan(log_ratios).any(), "output nan aborting"
        return log_ratios.mean(0)

    def entropy(self) -> torch.Tensor:
        """ H(self) estimated with monte carlo sampling
        """
        rsamples = self.__rsamples().unbind(0)
        log_ps = torch.stack([-self.logp(rsample) for rsample in rsamples])
        assert not torch.isnan(log_ps).any(), "output nan aborting"
        return log_ps.mean(0)

    def sample(self):
        normal_samples = self.normals.sample() # batch_size x num_gaussians x action_dim
        cat_samples = self.cat.sample() # batch_size
        # First we need to expand cat so that it has the same dimension as normal samples
        cat_samples = cat_samples.view(-1, 1, 1).expand(-1, -1, self.action_dim)
        # We select the normal distribution based on the outputs of 
        # the categorical distribution
        self.last_sample = torch.gather(normal_samples, 1, cat_samples).squeeze(dim=1) # batch_size x action_dim
        assert len(self.last_sample.shape) == 2, f"shape, {self.last_sample.shape}"
        return self.last_sample

ModelCatalog.register_custom_action_dist("gmm", TorchGaussianMixtureDistribution)

class TorchFlowDistribution(TorchDistributionWrapper):
    # https://github.com/ray-project/ray/blob/be62444bc5924c61d69bb6aec62f967e531e768c/rllib/examples/models/autoregressive_action_dist.py
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return prod(action_space.shape)

    def __init__(self, inputs: torch.Tensor, model: NormalizingFlowsPolicy):
        super(TorchDistributionWrapper, self).__init__(inputs, model)
        self.model = model
        self.batch_size, self.action_dim = inputs.shape
        self.device = inputs.device
        self.base_dist = Normal(
                torch.zeros(self.batch_size, self.action_dim, device=self.device), 
                torch.ones(self.batch_size, self.action_dim, device=self.device))
        self.monte_samples = model.model_config['custom_model_config']['monte_samples']
        self.zero = torch.zeros(self.batch_size, self.action_dim, device=self.device)
    
    def logp(self, actions: torch.Tensor) -> torch.Tensor:
        z, log_abs_det = self.model.flow(actions, context=self.inputs, inverse_mode=True)
        logp = self.base_dist.log_prob(z).sum(-1)
        return logp + log_abs_det

    def deterministic_sample(self) -> torch.Tensor:
        self.last_sample, _ = self.model.flow(self.zero, context=self.inputs)
        return self.last_sample

    def sample(self):
        z = self.base_dist.sample()
        self.last_sample, _ = self.model.flow(z, context=self.inputs)
        return self.last_sample

    def __rsamples_logps(self):
        base_samples = self.base_dist.sample((self.monte_samples,))
        logp_base_samples = self.base_dist.log_prob(base_samples).sum(-1)
        rsamples, log_abs_det = self.model.flow(base_samples, context=self.inputs.unsqueeze(0))
        logps = logp_base_samples - log_abs_det
        return rsamples, logps 

    def kl(self, q: ActionDistribution) -> torch.Tensor:
        """ KL(self || q) estimated with monte carlo sampling
        """
        rsamples, logps = self.__rsamples_logps()
        logp_rsamples = zip(logps.unbind(0), rsamples.unbind(0))
        log_ratios = torch.stack([q.logp(rsample) - logp for (logp, rsample) in logp_rsamples])
        assert not torch.isnan(log_ratios).any(), "output nan aborting"
        return log_ratios.mean(0)

    def entropy(self) -> torch.Tensor:
        """ H(self) estimated with monte carlo sampling
        """
        _, logps = self.__rsamples_logps()
        assert not torch.isnan(logps).any(), "output nan aborting" 
        return -logps.mean(0)

ModelCatalog.register_custom_action_dist("flow_dist", TorchFlowDistribution)
