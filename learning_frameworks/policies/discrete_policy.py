from typing import Tuple

import torch

from learning_frameworks.policies.policy import Policy


class DiscretePolicy(Policy):

    @property
    def n_actions(self):
        return 1

    def forward(self, input_) -> torch.Tensor:
        return torch.softmax(super().forward(input_), dim=-1)

    def act(self, input_, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fw_result = self.forward(input_)

        n_inputs = fw_result.shape[-1]

        if deterministic:
            return torch.argmax(fw_result, dim=-1), torch.zeros(size=(n_inputs, 1)), torch.zeros(size=(n_inputs, 1))
        else:
            distribution = torch.distributions.Categorical(fw_result)
            actions = distribution.sample()
            return actions, distribution.entropy(), distribution.log_prob(actions)

    def get_backprop_data(self, observations, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        fw_result = self.forward(observations)
        distribution = torch.distributions.Categorical(fw_result)

        return distribution.entropy(), distribution.log_prob(actions)