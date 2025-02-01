import torch

from learning_frameworks.action_formatters.base_action_formatter import BaseActionFormatter

class IntActionFormatter(BaseActionFormatter):
    """
    Formats actions to integers
    """
    def format_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.to(dtype=torch.int)

class FloatActionFormatter(BaseActionFormatter):
    """
    Format actions to floats
    """
    def format_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.to(dtype=torch.float)
