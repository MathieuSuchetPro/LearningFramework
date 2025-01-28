import json
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class Policy(nn.Module):
    POLICY_FILE: str = "policy.pth"
    CRITIC_FILE: str = "critic.pth"
    POLICY_OPTIMIZER_FILE: str = "policy_optimizer.pth"
    CRITIC_OPTIMIZER_FILE: str = "critic_optimizer.pth"
    MISC_DATA_FILE: str = "misc.json"

    def __init__(self, input_size: int, output_size: int, actor_layer_sizes: List[int], critic_layer_sizes: List[int],
                 actor_lr: float, critic_lr: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        actor_layers = [
            nn.Linear(input_size, actor_layer_sizes[0]),
            nn.ReLU()
        ]

        for i in range(1, len(actor_layers)):
            actor_layers.extend(
                [
                    nn.Linear(actor_layer_sizes[i - 1], actor_layer_sizes[i]),
                    nn.ReLU()
                ]
            )

        actor_layers.extend(
            [
                nn.Linear(actor_layer_sizes[-1], output_size)
            ]
        )

        critic_layers = [
            nn.Linear(input_size, critic_layer_sizes[0]),
            nn.ReLU()
        ]

        for i in range(1, len(critic_layers)):
            critic_layers.extend(
                [
                    nn.Linear(critic_layer_sizes[i - 1], critic_layer_sizes[i]),
                    nn.ReLU()
                ]
            )

        critic_layers.extend(
            [
                nn.Linear(critic_layer_sizes[-1], 1)
            ]
        )

        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

        self.input_size = input_size
        self.output_size = output_size

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.learning_step = 0

    def actor_forward(self, input_) -> torch.Tensor:
        if isinstance(input_, np.ndarray):
            input_ = torch.tensor(input_, dtype=torch.float)
        elif isinstance(input_, (List, Tuple)):
            input_ = torch.tensor(input_, dtype=torch.float).unsqueeze(0)

        return self.actor(input_)

    def critic_forward(self, input_) -> torch.Tensor:
        if isinstance(input_, (np.ndarray, List, Tuple)):
            input_ = torch.tensor(input_, dtype=torch.float)

        return self.critic(input_)

    def forward(self, input_):
        return self.actor_forward(input_), self.critic_forward(input_)

    @abstractmethod
    def act(self, input_, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns an action, entropy and log prob based on input
        :param input_: Input to use
        :param deterministic: Deterministic action or not
        :return: The action taken by the policy
        """
        pass

    @abstractmethod
    def get_backprop_data(self, observations, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the data necessary for backpropagation
        :param observations: Observations used to create the distribution
        :param actions: Actions to calculate log prob
        :return: Entropies, Log probs and values
        """
        pass

    def save(self, path: str):
        if "." in path:
            raise ValueError("Path contains \".\", it should be a directory name")

        path_ = pathlib.Path(os.getcwd()) / pathlib.Path(path)
        if not os.path.exists(path_):
            os.makedirs(path_)

        torch.save(self.actor.state_dict(), path_ / Policy.POLICY_FILE)
        torch.save(self.critic.state_dict(), path_ / Policy.CRITIC_FILE)

        torch.save(self.actor_optimizer.state_dict(), path_ / Policy.POLICY_OPTIMIZER_FILE)
        torch.save(self.critic_optimizer.state_dict(), path_ / Policy.CRITIC_OPTIMIZER_FILE)

        with open(path_ / Policy.MISC_DATA_FILE, "w") as f:
            json.dump({
                "learning_timesteps": self.learning_step
            }, f)

    def load(self, path: str):
        if "." in path:
            raise ValueError("Path contains \".\", it should be a directory name")

        path_ = pathlib.Path(os.getcwd()) / pathlib.Path(path)

        with open(path_ / Policy.MISC_DATA_FILE, "r") as f:
            misc = json.load(f)

        self.learning_step = misc["learning_timesteps"]

        self.actor.load_state_dict(torch.load(path_ / Policy.POLICY_FILE, weights_only=True))
        self.critic.load_state_dict(torch.load(path_ / Policy.CRITIC_FILE, weights_only=True))
        try:
            self.actor_optimizer.load_state_dict(torch.load(path_ / Policy.POLICY_OPTIMIZER_FILE, weights_only=True))
            self.critic_optimizer.load_state_dict(
                torch.load(path_ / Policy.CRITIC_OPTIMIZER_FILE, weights_only=True))
        except FileNotFoundError:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters())



    @property
    @abstractmethod
    def n_actions(self):
        pass