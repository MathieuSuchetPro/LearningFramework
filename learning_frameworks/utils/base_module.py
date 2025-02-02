import json
import os
import pathlib
from abc import abstractmethod
from typing import Callable, List, Tuple

import numpy as np
import torch.nn
from torch import nn


class BaseModule(nn.Module):
    NN_FILE: str = "net.pth"
    OPTIMIZER_FILE: str = "optim.pth"
    MISC_FILE: str = "misc.json"

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_sizes: List[int],
            activation_fn: Callable[[], torch.nn.Module],

            learning_rate: float,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.layer_sizes = layer_sizes
        self.output_size = output_size
        self.input_size = input_size

        layers = [
            nn.Linear(input_size, layer_sizes[0]),
            self.activation_fn()
        ]

        for i in range(1, len(layer_sizes)):
            layers.extend(
                [
                    nn.Linear(layer_sizes[i - 1], layer_sizes[i]),
                    self.activation_fn()
                ]
            )

        layers.extend(
            [
                nn.Linear(layer_sizes[-1], 1)
            ]
        )

        self.nn = torch.nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        self.learning_step = 0

    def forward(self, input_):
        if isinstance(input_, np.ndarray):
            input_ = torch.tensor(input_, dtype=torch.float)
        elif isinstance(input_, (List, Tuple)):
            input_ = torch.tensor(input_, dtype=torch.float).unsqueeze(0)

        return self.nn(input_)

    def save(self, path: pathlib.Path):
        if path.is_file():
            raise ValueError("Path contains \".\", it should be a directory name")

        path_ = pathlib.Path(os.getcwd()) / pathlib.Path(path)

        if not os.path.exists(path_):
            os.makedirs(path_)

        torch.save(self.nn.state_dict(), path_ / BaseModule.NN_FILE)
        torch.save(self.optimizer.state_dict(), path_ / BaseModule.OPTIMIZER_FILE)

        with open(path_ / BaseModule.MISC_FILE, "w") as f:
            json.dump({
                "learning_timesteps": self.learning_step
            }, f)

    def load(self, path: pathlib.Path):
        if path.is_file():
            raise ValueError("Path contains \".\", it should be a directory name")

        path_ = pathlib.Path(os.getcwd()) / path

        with open(path_ / BaseModule.MISC_FILE, "r") as f:
            misc = json.load(f)

        self.learning_step = misc["learning_timesteps"]

        self.nn.load_state_dict(torch.load(path_ / BaseModule.NN_FILE, weights_only=True))
        try:
            self.optimizer.load_state_dict(
                torch.load(path_ / BaseModule.OPTIMIZER_FILE, weights_only=True))
        except FileNotFoundError:
            self.optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.learning_rate)

    @abstractmethod
    def get_backprop_data(self, observations, actions):
        """
        Get the data necessary for backpropagation
        :param observations: Observations used to create the distribution
        :param actions: Actions to calculate log prob
        :return: Entropies, Log probs
        """
        pass