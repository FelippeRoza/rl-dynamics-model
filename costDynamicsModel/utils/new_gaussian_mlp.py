import mbrl
from mbrl.models.gaussian_mlp import GaussianMLP

import torch
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import omegaconf

class GaussianMLPNew(GaussianMLP):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        super(GaussianMLPNew, self).__init__(in_size = in_size,
            out_size = out_size,
            device = device,
            num_layers = num_layers,
            ensemble_size = ensemble_size,
            hid_size = hid_size,
            deterministic = deterministic,
            propagation_method = propagation_method,
            learn_logvar_bounds= learn_logvar_bounds,
            activation_fn_cfg = activation_fn_cfg,
        )
 
    def set_shapes(self, obs_shape, action_shape, cost_shape):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.cost_shape = cost_shape
    
    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, return_g: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)

        x = x.float()
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            actions = x[:, -self.action_shape[0] :].clone().detach()
            actions = actions.repeat(self.num_members, 1, 1)
            # x[:, -self.action_shape[0] :] = 1.0
            # x = x[:, :-self.action_shape[0]]
        else:
            actions = x[:, :, -self.action_shape[0]:].clone().detach()
            # x[:, :, -self.action_shape[0] :] = 0
        
            
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        if self.deterministic:

            mean_and_logvar = mean_and_logvar.reshape((mean_and_logvar.shape[0], mean_and_logvar.shape[1], 
                                                       self.cost_shape[0], self.action_shape[0]))
            if return_g:
                return mean_and_logvar, None

            mean_and_logvar = torch.matmul(mean_and_logvar, actions.unsqueeze(-1)).squeeze(-1)
            
            return mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., : self.out_size]
            logvar = mean_and_logvar[..., self.out_size :]
            if return_g:
                return mean.reshape((mean.shape[0], mean.shape[1], self.cost_shape[0], self.action_shape[0])), None
            
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

            mean = mean.reshape((mean.shape[0], mean.shape[1], self.cost_shape[0], self.action_shape[0]))
            mean = torch.matmul(mean, actions.unsqueeze(-1)).squeeze(-1)

            logvar = logvar.reshape((logvar.shape[0], logvar.shape[1], self.cost_shape[0], self.action_shape[0]))
            logvar = torch.matmul(logvar, actions.unsqueeze(-1)).squeeze(-1)
            return mean, logvar
    
    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # actions = model_in[:, :, self.obs_shape[0] + self.action_shape[0]:]
        # model_in[:, :, self.obs_shape[0] + self.action_shape[0]:] = 0
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll