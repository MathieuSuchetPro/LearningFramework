from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor

from learning_frameworks.collection.buffer import BaseBuffer
from learning_frameworks.policies.policy import Policy
from learning_frameworks.value_estimators.value_estimator import ValueEstimator


class Agent(ABC):
    def __init__(self, policy: Policy, value_estimator: ValueEstimator):
        self.value_estimator = value_estimator
        self.policy = policy

    def act(self, input_, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return the action taken by the policy
        :param input_: Input to take action on
        :param deterministic: Whether the action is deterministic
        :return: Actions, Entropies and Log probs
        """
        return self.policy.act(input_, deterministic)

    def estimate_values(self, input_) -> Tensor:
        """
        Estimates the value of the inputs
        :param input_: Input to estimate the value of
        :return: The estimated values
        """
        return self.value_estimator(input_)

    @abstractmethod
    def learn(self, buffer: BaseBuffer):
        pass

