import multiprocessing as mp
import queue
from typing import Tuple

import numpy as np
import torch
import tqdm

from learning_frameworks.collection.buffer import Trajectory, StepResult, AgentResult, BaseBuffer
from learning_frameworks.learning.agent import Agent
from learning_frameworks.vectorization.env_process import _process


class ProcessManager:
    def __init__(self, n_proc: int, env_create_fn, agent: Agent, buffer: BaseBuffer):
        self.action_size = agent.policy.n_actions
        self.obs_size = agent.policy.input_size
        self.n_proc = n_proc
        self.buffer = buffer

        self.agent = agent

        self.procs = []
        self.to_parent_remotes = []
        self.to_proc_remotes = []
        self.trajectories = []

        self._last_obs = torch.FloatTensor(size=(n_proc, self.obs_size))

        self.queue = queue.Queue()

        process_init_bar = tqdm.tqdm(desc="Processes init", total=n_proc)

        for i in range(n_proc):
            to_child_pipe, to_parent_pipe = mp.Pipe()

            # Create the process
            proc = mp.Process(target=_process, args=(env_create_fn(), to_parent_pipe))
            process_init_bar.update(1)

            self.procs.append(proc)
            self.to_proc_remotes.append(to_child_pipe)
            self.to_parent_remotes.append(to_parent_pipe)
            self.trajectories.append(Trajectory(16_384, self.obs_size, self.action_size))

    def start(self):
        process_start_bar = tqdm.tqdm(desc="Processes start", total=self.n_proc)

        for proc in self.procs:
            proc.start()
            process_start_bar.update(1)

    def reset(self):
        for to_proc_remote in self.to_proc_remotes:
            to_proc_remote.send({"name": "reset"})

        all_obs = torch.FloatTensor(torch.zeros(size=(self.n_proc, self.obs_size)))
        all_info = []

        for i, to_proc_remote in enumerate(self.to_proc_remotes):
            obs, info = to_proc_remote.recv()

            all_obs[i] = torch.tensor(obs, dtype=torch.float)
            all_info.append(info)

        self._last_obs = all_obs

        return all_obs, np.asarray(all_info)

    def step(self, agent_result: AgentResult) -> Tuple[StepResult, torch.Tensor]:
        n_actions = np.shape(agent_result.actions)[0]
        if n_actions != len(self.procs):
            raise IndexError(f"Expecting {len(self.procs)} actions, got {n_actions} actions")

        all_next_obs = torch.zeros(size=(self.n_proc, self.obs_size), dtype=torch.float)
        all_obs = torch.zeros(size=(self.n_proc, self.obs_size), dtype=torch.float)
        all_rewards = torch.zeros(size=(self.n_proc, 1), dtype=torch.float)
        all_done = torch.zeros(size=(self.n_proc, 1), dtype=torch.short)

        all_reset_next_obs = torch.zeros(size=(self.n_proc, self.obs_size), dtype=torch.float)

        for i, to_proc_remote in enumerate(self.to_proc_remotes):
            to_proc_remote.send({"name": "step", "actions": agent_result.actions[i].detach().numpy()})

        for i, to_proc_remote in enumerate(self.to_proc_remotes):
            next_obs, r, terminal, truncated, info, reset_next_obs, reset_info = to_proc_remote.recv()

            next_obs = torch.tensor(next_obs, dtype=torch.float)
            r = torch.tensor([r], dtype=torch.float)
            terminal = torch.tensor([terminal], dtype=torch.bool)
            truncated = torch.tensor([truncated], dtype=torch.bool)
            reset_next_obs = torch.tensor(reset_next_obs, dtype=torch.float)

            done = torch.logical_or(terminal, truncated)

            if truncated:
                r = self.agent.value_estimator(next_obs).detach()

            all_obs[i] = self._last_obs[i]
            all_rewards[i] = r
            all_done[i] = done
            all_next_obs[i] = next_obs

            all_reset_next_obs[i] = reset_next_obs

            current_step_result = StepResult(
                observations=all_obs[i],
                rewards=all_rewards[i],
                done=all_done[i],
                next_observations=all_next_obs[i]
            )

            self.trajectories[i].add(current_step_result,
                                     AgentResult(
                                         actions=agent_result.actions[i],
                                         entropies=agent_result.entropies[i],
                                         log_probs=agent_result.log_probs[i]
                                     ))

            if done:
                self.buffer.concat(self.trajectories[i])
                self.trajectories[i].clear()

        self._last_obs = all_reset_next_obs

        return StepResult(
            observations=all_obs,
            rewards=all_rewards,
            done=all_done,
            next_observations=all_next_obs
        ), all_reset_next_obs

    def get_env_metrics(self):
        for to_proc_remote in self.to_proc_remotes:
            to_proc_remote.send({"name": "get_metrics"})

        all_metrics = []
        for to_proc_remote in self.to_proc_remotes:
            metrics = to_proc_remote.recv()
            all_metrics.append(metrics)

        return np.asarray(all_metrics)

    def close(self):
        for to_proc_remote in self.to_proc_remotes:
            to_proc_remote.send({"name": "close"})
            to_proc_remote.close()
