from abc import abstractmethod
from typing import NamedTuple, Tuple, Any, Union, List

import numpy as np
import torch
from numpy import ndarray, dtype


class StepResult(NamedTuple):
    observations: torch.FloatTensor
    rewards: torch.FloatTensor
    done: torch.ShortTensor
    next_observations: torch.FloatTensor


class AgentResult(NamedTuple):
    actions: torch.FloatTensor
    entropies: torch.FloatTensor
    log_probs: torch.FloatTensor


class BaseBuffer:
    """
    The base class for the buffer
    """

    def __init__(self, buffer_size, observation_size, action_size):
        """
        :param buffer_size: Size of the buffer
        :param observation_size: Size of an observation
        :param action_size: Size of an action
        """
        self.buffer_size = buffer_size
        self.observation_size = observation_size
        self.actions_size = action_size
        self.cnt_i = 0
        self.value_i = 0

        (self.states, self.actions, self.rewards, self.dones, self.next_states,
         self.values, self.entropies, self.log_probs) = self.init_buffers()

    @abstractmethod
    def init_buffers(self) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.ShortTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor
    ]:
        """
        Allocates the memory for the buffers
        :return: All the empty buffers
        """
        pass

    @abstractmethod
    def full(self) -> bool:
        """
        Whether the buffers are full
        :return: Whether the buffers are full
        """
        pass

    @abstractmethod
    def get_batches(self, batch_size: int) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.ShortTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.IntTensor
    ]:
        """
        Separate the buffers into batches
        :param batch_size: Size of a batch
        :return: States, Actions, Rewards, Dones, Values, Entropies, Log Probs, Batches indices
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Empties the buffers
        """
        pass

    @abstractmethod
    def add(self, step_result: StepResult, agent_result: AgentResult) -> None:
        """
        Adds an experience to the buffers
        :param step_result: Result of the step
        :param agent_result: Result of the agent on that step
        """
        pass

    @abstractmethod
    def add_values(self, values: torch.FloatTensor) -> None:
        """
        Adds values to the value buffer
        :param values: Values to add
        """
        pass

    @abstractmethod
    def concat(self, buffer: "BaseBuffer") -> None:
        """
        Concatenates a buffer to the current buffer
        :param buffer: The buffer to add
        """
        pass


class Trajectory(BaseBuffer):
    def concat(self, buffer: "BaseBuffer") -> None:
        pass

    def init_buffers(self) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.ShortTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor
    ]:
        states = torch.FloatTensor(size=(self.buffer_size, self.observation_size))
        actions = torch.FloatTensor(size=(self.buffer_size, self.actions_size))
        rewards = torch.FloatTensor(size=(self.buffer_size, 1))
        dones = torch.ShortTensor(size=(self.buffer_size, 1))
        next_states = torch.FloatTensor(size=(self.buffer_size, self.observation_size))
        values = torch.FloatTensor(size=(self.buffer_size, 1))
        entropies = torch.FloatTensor(size=(self.buffer_size, 1))
        log_probs = torch.FloatTensor(size=(self.buffer_size, 1))

        return states, actions, rewards, dones, next_states, values, entropies, log_probs

    def full(self) -> bool:
        return self.cnt_i >= self.buffer_size

    def clear(self) -> None:
        self.states.fill_(0)
        self.actions.fill_(0)
        self.rewards.fill_(0)
        self.dones.fill_(0)
        self.next_states.fill_(0)
        self.values.fill_(0)
        self.entropies.fill_(0)
        self.log_probs.fill_(0)

        self.cnt_i = 0
        self.value_i = 0

    def get_batches(self, batch_size: int) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.ShortTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.IntTensor
    ]:
        pass

    def add_values(self, values: torch.FloatTensor) -> None:
        values_ = values.squeeze()
        self.values[self.value_i:self.value_i + values.shape[0]] = values_
        self.value_i += values_.shape[0]

    def add(self, step_result: StepResult, agent_result: AgentResult) -> None:
        if self.full():
            return

        self.states[self.cnt_i] = step_result.observations.clone()
        self.actions[self.cnt_i] = agent_result.actions.clone()
        self.rewards[self.cnt_i] = step_result.rewards.clone()
        self.dones[self.cnt_i] = step_result.done.clone()
        self.next_states[self.cnt_i] = step_result.next_observations.clone()
        self.entropies[self.cnt_i] = agent_result.entropies.clone()
        self.log_probs[self.cnt_i] = agent_result.log_probs.clone()

        self.cnt_i += 1


class Buffer(BaseBuffer):
    def init_buffers(self) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.ShortTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor
    ]:
        states = torch.FloatTensor(size=(self.buffer_size, self.observation_size))
        actions = torch.FloatTensor(size=(self.buffer_size, self.actions_size))
        rewards = torch.FloatTensor(size=(self.buffer_size, 1))
        dones = torch.ShortTensor(size=(self.buffer_size, 1))
        next_states = torch.FloatTensor(size=(self.buffer_size, self.observation_size))
        values = torch.FloatTensor(size=(self.buffer_size, 1))
        entropies = torch.FloatTensor(size=(self.buffer_size, 1))
        log_probs = torch.FloatTensor(size=(self.buffer_size, 1))

        return states, actions, rewards, dones, next_states, values, entropies, log_probs

    def full(self) -> bool:
        return self.cnt_i >= self.buffer_size

    def clear(self):
        self.states.fill_(0)
        self.actions.fill_(0)
        self.rewards.fill_(0)
        self.dones.fill_(0)
        self.next_states.fill_(0)
        self.values.fill_(0)
        self.entropies.fill_(0)
        self.log_probs.fill_(0)

        self.cnt_i = 0
        self.value_i = 0

    def get_batches(self, batch_size: int) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.ShortTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        List
    ]:
        batch_start = torch.arange(0, self.cnt_i, batch_size)
        indices = torch.arange(0, self.cnt_i, 1)
        np.random.shuffle(indices.numpy())
        batches = [indices[i:i + batch_size] for i in batch_start]

        return self.states, self.actions, self.rewards, self.dones, self.values, self.entropies, self.log_probs, batches

    def add(self, step_result: StepResult, agent_result: AgentResult):
        pass

    def concat(self, buffer: BaseBuffer):
        end_i = self.buffer_size - self.cnt_i
        end_i = min(end_i, buffer.cnt_i)

        self.states[self.cnt_i:self.cnt_i + end_i] = buffer.states[:end_i].clone()
        self.actions[self.cnt_i:self.cnt_i + end_i] = buffer.actions[:end_i].clone()
        self.rewards[self.cnt_i:self.cnt_i + end_i] = buffer.rewards[:end_i].clone()
        self.dones[self.cnt_i:self.cnt_i + end_i] = buffer.dones[:end_i].clone()
        self.next_states[self.cnt_i:self.cnt_i + end_i] = buffer.next_states[:end_i].clone()
        self.values[self.cnt_i:self.cnt_i + end_i] = buffer.values[:end_i].clone()
        self.entropies[self.cnt_i:self.cnt_i + end_i] = buffer.entropies[:end_i].clone()
        self.log_probs[self.cnt_i:self.cnt_i + end_i] = buffer.log_probs[:end_i].clone()

        self.cnt_i += end_i

    def add_values(self, values: torch.FloatTensor):
        self.values[self.value_i:self.value_i + values.shape[0]] = values
        self.value_i += values.shape[0]
