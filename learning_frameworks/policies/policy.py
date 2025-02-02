from abc import abstractmethod
from typing import Tuple

import torch

from learning_frameworks.utils.base_module import BaseModule


class Policy(BaseModule):
    @abstractmethod
    def act(self, input_, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns an action, entropy and log prob based on input
        :param input_: Input to use
        :param deterministic: Deterministic action or not
        :return: The action taken by the policy
        """
        pass


    @property
    @abstractmethod
    def n_actions(self):
        pass