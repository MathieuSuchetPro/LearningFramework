from typing import Tuple, List

import torch
from torch import nn

from learning_frameworks.policies.policy import Policy

class MapContinuousToAction(nn.Module):
    """
    A class for policies using the continuous action space. Continuous policies output N*2 values for N actions where
    each value is in the range [-1, 1]. Half of these values will be used as the mean of a multi-variate normal distribution
    and the other half will be used as the diagonal of the covariance matrix for that distribution. Since variance must
    be positive, this class will map the range [-1, 1] for those values to the desired range (defaults to [0.1, 1]) using
    a simple linear transform.
    """
    def __init__(self, range_min=0.1, range_max=1):
        super().__init__()

        tanh_range = [-1, 1]
        self.m = (range_max - range_min) / (tanh_range[1] - tanh_range[0])
        self.b = range_min - tanh_range[0] * self.m

    def forward(self, x):
        n = x.shape[-1] // 2
        # map the right half of x from [-1, 1] to [range_min, range_max].
        return x[..., :n], x[..., n:] * self.m + self.b

class ContinuousPolicy(Policy):
    def __init__(self, input_size: int, output_size: int, actor_layer_sizes: List[int], critic_layer_sizes: List[int],
                 actor_lr: float, critic_lr: float, var_min: float = 0.1, var_max: int = 1, *args, **kwargs):
        super().__init__(input_size, output_size * 2, actor_layer_sizes, critic_layer_sizes, actor_lr, critic_lr, *args,
                         **kwargs)
        self.var_min = var_min
        self.var_max = var_max

        self.map_params = MapContinuousToAction(self.var_min, self.var_max)

    def actor_forward(self, input_) -> torch.Tensor:
        return self.map_params(torch.tanh(super().actor_forward(input_)))

    def act(self, input_, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fw_mean, fw_std = self.actor_forward(input_)

        if deterministic:
            return fw_mean, torch.zeros(size=(self.output_size // 2, 1)), torch.ones(size=(self.output_size // 2, 1))
        else:
            distrib = torch.distributions.Normal(loc=fw_mean, scale=fw_std)
            actions = distrib.sample()
            return actions, distrib.entropy().sum(-1), distrib.log_prob(actions).sum(-1)



    def get_backprop_data(self, observations, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fw_mean, fw_std = self.actor_forward(observations)
        distrib = torch.distributions.Normal(loc=fw_mean, scale=fw_std)

        return distrib.entropy().sum(-1), distrib.log_prob(actions).sum(-1), self.critic_forward(observations)

    @property
    def n_actions(self):
        return self.output_size // 2