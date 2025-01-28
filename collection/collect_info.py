from typing import Callable

import numpy as np
import torch
from gymnasium import Env
from tqdm import tqdm

from TenthTry.agent import Agent
from TenthTry.buffer import Buffer, AgentResult, StepResult
from TenthTry.vectorization.process_manager import ProcessManager


def collect_buffer(buffer: Buffer, agent: Agent, env_fn: Callable[[], Env], n_proc: int):
    proc_manager = ProcessManager(
        n_proc=n_proc,
        env_create_fn=env_fn
    )

    proc_manager.start()

    progress_bar = tqdm(total=buffer.buffer_size * n_proc, desc=f"Iteration {agent.learning_step}")

    while True:
        try:
            obs, _ = proc_manager.reset()
            while not buffer.full():
                actions, entropies, log_probs = agent.act(torch.tensor(obs, dtype=torch.float), deterministic=False)
                actions = actions.numpy()
                entropies = entropies.detach().numpy()
                log_probs = log_probs.detach().numpy()

                next_obs, rewards, terminal, truncated, info, reset_next_obs, reset_info = proc_manager.step(actions)

                if rewards[truncated].size > 0:
                    truncated_vals = agent.critic_forward(torch.tensor(obs[truncated], dtype=torch.float))
                    rewards[truncated] = torch.squeeze(truncated_vals).detach().cpu().numpy()

                s = StepResult(
                    observations=obs,
                    rewards=rewards,
                    done=np.bitwise_or(terminal, truncated),
                    next_observations=reset_next_obs
                )

                if len(actions.shape) == 1:
                    actions = actions.reshape(n_proc, 1)
                    entropies = entropies.reshape(n_proc, 1)
                    log_probs = log_probs.reshape(n_proc, 1)

                agent_result = AgentResult(actions=actions, entropies=entropies, log_probs=log_probs)

                buffer.add(s, agent_result)
                progress_bar.update(n_proc)

                obs = reset_next_obs

            values = agent.critic_forward(
                torch.tensor(buffer.states.reshape((buffer.buffer_size * n_proc, agent.input_size), order="F")))
            buffer.add_values(values.detach().numpy())

            progress_bar.close()

            return proc_manager.get_env_metrics()

        except KeyboardInterrupt:
            proc_manager.close()
            raise KeyboardInterrupt("User interruption during buffer collection")


def render_loop(agent: Agent, env_fn: Callable[[], Env]):
    env = env_fn()
    while True:
        obs, _ = env.reset()

        terminated, truncated = False, False

        while not terminated and not truncated:

            env.render()

            actions, _, _ = agent.act(torch.tensor(obs, dtype=torch.float), deterministic=False)
            actions = actions.numpy()

            next_obs, _, terminated, truncated, _ = env.step(actions)

            obs = next_obs