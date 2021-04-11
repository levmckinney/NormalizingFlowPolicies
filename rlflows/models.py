from logging import NOTSET
from typing import List
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

class FlowLayer(nn.Module):
    def __init__(self, dimension: int):
        """
            Args:
                dimension: the dimension of the flow
        """
        super().__init__()
        self.dimension = dimension

    def forward(self, x, inverse_mode=False, context=None):
        """ Transform the variable x to f and return the transformations
            log_abs_det_jac f. If in inverse mode return x transformed according to f^-1
            and return log_abs_det_jac f^-1.
            Layers may choose to condition on context.
        """
        ...

class ConditioningLayer(FlowLayer):
    """ Allows for the injection of additional context into a flow. 
    """
    def forward(self, x, inverse_mode=False, context=None):
        log_abs_det = x[..., 0]*0
        if context == None:
            return x, log_abs_det
        if inverse_mode:
            return x - context, log_abs_det
        else:
            return x + context, log_abs_det

class CouplingLayer(FlowLayer):
    """ Implements RealNVP based coupling layer
    """
    def __init__(self, dimension: int, hidden_size: int, hidden_layers: int, flipped: bool = False):
        super().__init__(dimension)
        assert dimension >= 2
        self.ident_size =  (dimension + flipped)//2
        self.trans_size = dimension - self.ident_size
        self.flipped = flipped
        self.s = MLP(self.ident_size, self.trans_size, hidden_size, hidden_layers)
        self.t = MLP(self.ident_size, self.trans_size, hidden_size, hidden_layers)
    
    def _pack(self, ident, trans):
        x = torch.cat((ident, trans), dim=-1)
        if self.flipped:
            return x.flip(dims=(-1,))
        else:
            return x

    def _unpack(self, x):
        if self.flipped:
            x = x.flip(dims=(-1,))
        return x[..., :self.ident_size], x[..., self.ident_size:]

    def forward(self, x, inverse_mode=False, context=None):
        ident, trans = self._unpack(x)
        s = self.s(ident)
        t = self.t(ident)
        if inverse_mode:
            assert trans.shape == t.shape, f"{t.shape}, {trans.shape}, {x.shape}"
            trans = (trans - t) * torch.exp(-s)
            log_abs_det = torch.sum(s, dim=-1)
        else:
            trans = trans * torch.exp(s) + t
            log_abs_det = -torch.sum(s, dim=-1)
        x = self._pack(ident, trans)
        return x, log_abs_det

class ComposeFlows(FlowLayer):
    def __init__(self, flows: List[FlowLayer]):
        assert len(flows) > 0
        dimension = flows[0].dimension
        for flow in flows:
            assert flow.dimension == dimension
        super().__init__(dimension)
        self.flows = nn.ModuleList(flows)

    def forward(self, x, inverse_mode=False, context=None):
        flows = reversed(self.flows) if inverse_mode else self.flows
        total_log_abs_det = None
        for flow in flows:
            x, log_abs_det = flow(x, inverse_mode, context)
            if total_log_abs_det is None:
                total_log_abs_det = log_abs_det
            else:
                total_log_abs_det += log_abs_det
        return x, total_log_abs_det

class ConditionalCouplingFlow(FlowLayer):
    def __init__(self, dimension: int, flow_layers: int, hidden_size: int, hidden_layers: int, inject_after: int=1):
        super().__init__(dimension)
        layers = []
        for i in range(flow_layers):
            layers.append(
                CouplingLayer(dimension, hidden_size, hidden_layers, flipped=(i % 2 == 0)))
            if i + 1 == inject_after:
                layers.append(
                    ConditioningLayer(dimension))
        self.inner = ComposeFlows(layers)

    def forward(self, x, inverse_mode=False, context=None):
        return self.inner(x, inverse_mode, context)

class NormalizingFlowsPolicy(FullyConnectedNetwork):
    # https://github.com/ray-project/ray/blob/be62444bc5924c61d69bb6aec62f967e531e768c/rllib/examples/models/autoregressive_action_model.py#L87
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.coupling_hidden_size = model_config['custom_model_config'].get('coupling_hidden_size')
        self.coupling_hidden_layers = model_config['custom_model_config'].get('coupling_hidden_layers')
        self.num_flow_layers = model_config['custom_model_config'].get('num_flow_layers')
        self.inject_state_after = model_config['custom_model_config'].get('inject_state_after')
        self.flow = ConditionalCouplingFlow(
            num_outputs, 
            self.num_flow_layers, 
            self.coupling_hidden_size,
            self.coupling_hidden_layers, 
            self.inject_state_after)

ModelCatalog.register_custom_model("flow_model", NormalizingFlowsPolicy)
