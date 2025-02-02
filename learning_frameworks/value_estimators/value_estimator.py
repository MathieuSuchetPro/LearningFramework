from abc import ABC

import torch

from learning_frameworks.utils.base_module import BaseModule


class ValueEstimator(BaseModule, ABC):
    def get_backprop_data(self, observations, actions) -> torch.Tensor:
        return self.forward(observations)