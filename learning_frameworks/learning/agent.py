from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor

from learning_frameworks.collection.buffer import BaseBuffer
from learning_frameworks.policies.policy import Policy


class Agent(ABC):
    def __init__(self, policy: Policy):
        self.policy = policy

    def act(self, input_, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return the action taken by the policy
        :param input_: Input to take action on
        :param deterministic: Whether the action is deterministic
        :return: Actions, Entropies and Log probs
        """
        return self.policy.act(input_, deterministic)

    @abstractmethod
    def learn(self, buffer: BaseBuffer):
        pass

