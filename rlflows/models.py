from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from torch.distributions.transforms import Transform, ComposeTransform
from torch.distributions import constraints
from ray.rllib.utils.typing import ModelConfigDict

import gym
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=64, num_layers=2, activation=nn.ReLU) -> None:
        super().__init__()
        assert num_layers >= 1
        module_list = []
        module_list.append(nn.Linear(in_features, hidden_size))
        for _ in range(num_layers - 1):
            module_list.append(activation())
            module_list.append(nn.Linear(hidden_size, hidden_size))
        module_list.append(nn.Linear(hidden_size, out_features))
        self.base_model = nn.Sequential(*module_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

class CouplingLayer(Transform):
    # https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#AffineTransform
    # TODO look into pytorch cache for this network/use existing implementation
    bijective = True
    def __init__(self, dimension: int, flipped: bool, hidden_size: int, hidden_layers: int):
        super().__init__(cache_size=0)
        assert dimension >= 2
        self.ident_size =  (dimension + flipped)//2
        self.trans_size = dimension - self.ident_size
        self.flipped = flipped
        self.s = MLP(self.ident_size, self.trans_size, hidden_size, hidden_layers)
        self.t = MLP(self.ident_size, self.trans_size, hidden_size, hidden_layers)
    
    @property
    def codomain(self):
        return constraints.independent(constraints.real, 1)

    def _pack(self, ident, trans):
        z = torch.cat((ident, trans), dim=-1)
        if self.flipped:
            return z.flip(dims=(-1,))
        else:
            return z

    def _unpack(self, z):
        if self.flipped:
            z = z.flip(dims=(-1,))
        return z[..., :self.ident_size], z[..., self.ident_size:]

    def _call(self, x):
        ident, trans = self._unpack(x)
        trans = trans * torch.exp(self.s(ident)) + self.t(ident)
        return self._pack(ident, trans)

    def _inverse(self, y):
        ident, trans = self._unpack(y)
        trans = (trans - self.t(ident)) * torch.exp(-self.s(ident))
        return self._pack(ident, trans)
    
    def log_abs_det_jacobian(self, x, y):
        ident, _ = self._unpack(x)
        return torch.sum(self.s(ident), dim=-1)

class Injection(Transform):
    bijective=True
    def __init__(self, embeding: torch.Tensor):
        super().__init__(cache_size=0)
        self.embeding = embeding
    
    @property
    def event_dim(self):
        return 1

    @property
    def domain(self):
        return constraints.independent(constraints.real, 1)
    
    @property
    def codomain(self):
        return constraints.independent(constraints.real, 1)

    def _call(self, x):
        return x + self.embeding

    def _inverse(self, y):
        return y - self.embeding
    
    def log_abs_det_jacobian(self, x, y):
        return x[..., 0]*0

class NormalizingFlowsPolicy(FullyConnectedNetwork):
    # https://github.com/ray-project/ray/blob/be62444bc5924c61d69bb6aec62f967e531e768c/rllib/examples/models/autoregressive_action_model.py#L87
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.coupling_hidden_size = model_config['custom_model_config'].get('coupling_hidden_size')
        self.coupling_hidden_layers = model_config['custom_model_config'].get('coupling_hidden_layers')
        self.num_coupling_layers = model_config['custom_model_config'].get('num_coupling_layers')
        self.inject_state_after = model_config['custom_model_config'].get('inject_state_after')

        flow_layers = []
        identity_left = True
        for _ in range(self.num_coupling_layers):
            flow_layers.append(
                CouplingLayer(num_outputs, identity_left, self.coupling_hidden_size, self.coupling_hidden_layers)
            )
            identity_left = not identity_left
        self.flow_layers = torch.nn.ModuleList(flow_layers)
        print(self)

    def get_flow(self, state_embeding: torch.Tensor) -> ComposeTransform:
        return ComposeTransform(
            self.flow_layers[:self.inject_state_after] 
            + [Injection(state_embeding)] 
            + self.flow_layers[self.inject_state_after:])

ModelCatalog.register_custom_model("flow_model", NormalizingFlowsPolicy)