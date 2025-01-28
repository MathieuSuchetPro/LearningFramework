from abc import abstractmethod
from typing import Callable, Optional

import torch
from gymnasium import Env
from tqdm import tqdm

from learning_frameworks.collection.buffer import BaseBuffer, AgentResult
from learning_frameworks.collection.callbacks import Callback, EmptyCallback
from learning_frameworks.learning.PPO import PPO
from learning_frameworks.vectorization.process_manager import ProcessManager


# TODO: Make the metric collection system

class BaseCollection:
    """
    Base class for collection classes
    """

    def __init__(self, env_fn: Callable[[], Env], agent: PPO, callback: Optional[Callback], buffer: BaseBuffer):
        """
        :param env_fn: Function to create an environment
        :param agent: Agent to use
        :param callback: Callback
        :param buffer: Buffer to fill
        """
        self.env_fn = env_fn
        self.agent = agent
        self.callback = EmptyCallback() if not callback else callback
        self.buffer = buffer

    @abstractmethod
    def collect(self):
        """
        Fills the buffer with experience using the environment and the agent
        """
        pass

    def close(self):
        """
        Post run operations
        """
        pass


class Collection(BaseCollection):
    def __init__(self, env_fn: Callable[[], Env], agent: PPO, callback: Optional[Callback], buffer: BaseBuffer,
                 n_proc: int = 1):
        super().__init__(env_fn, agent, callback, buffer)
        self.n_proc = n_proc

        self.process_manager = ProcessManager(n_proc, env_fn, agent, buffer)
        self.process_manager.start()

    def collect(self):

        progress_bar = tqdm(total=self.buffer.buffer_size,
                                 desc=f"Iteration {self.agent.policy.learning_step}")

        self.callback.before_collection()
        progress_bar.set_description(f"Iteration {self.agent.policy.learning_step}")

        last_buffer_size = 0

        try:
            self.callback.before_collection()
            obs, _ = self.process_manager.reset()
            while not self.buffer.full():
                actions, entropies, log_probs = self.agent.act(obs, deterministic=False)
                actions = actions.to(dtype=torch.float)
                entropies = entropies.detach()
                log_probs = log_probs.detach()

                agent_result = AgentResult(actions=actions, entropies=entropies, log_probs=log_probs)

                self.callback.before_step(agent_result)

                step_result, reset_next_obs = self.process_manager.step(
                    agent_result)

                self.callback.after_step(step_result)

                obs = reset_next_obs

                progress_bar.update(self.buffer.cnt_i - last_buffer_size)
                last_buffer_size = self.buffer.cnt_i

            values = self.agent.policy.critic_forward(self.buffer.states)
            self.buffer.add_values(values.detach())

            progress_bar.close()
            self.callback.after_collection()

            return self.process_manager.get_env_metrics()

        except KeyboardInterrupt:
            self.close()
            raise KeyboardInterrupt("User interruption during buffer collection")

    def close(self):
        self.process_manager.close()
