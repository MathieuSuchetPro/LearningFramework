from abc import ABC, abstractmethod

import torch


class BaseActionFormatter(ABC):
    """
    A class that formats the policy's output to fit the environment constraints
    """
    @abstractmethod
    def format_actions(self, actions) -> torch.Tensor:
        """
        Format actions for the environment
        @param actions: Policy output
        @return: Formatted actions
        """
        pass