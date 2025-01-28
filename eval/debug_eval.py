from typing import Callable, Dict

import torch
from gymnasium import Env

from agent import Agent
from eval.eval import print_report


def critic_only_eval(
        env_fn: Callable[[], Env],
        agent: Agent,
        n_episodes: int = 5
) -> Dict[str, float]:
    env = env_fn()

    final_report = {
        "value_mean": torch.zeros(size=(n_episodes, 1)),
        "value_std": torch.zeros(size=(n_episodes, 1)),
        "value_min": torch.zeros(size=(n_episodes, 1)),
        "value_max": torch.zeros(size=(n_episodes, 1)),
    }

    all_obs = []

    for i in range(n_episodes):
        obs, _ = env.reset()

        terminated, truncated = False, False

        all_obs.append(obs.copy())

        while not terminated and not truncated:
            actions, _, _ = agent.act(obs, False)
            actions = actions.to(dtype=torch.float)

            next_obs, _, terminated, truncated, _ = env.step(actions.unsqueeze(0))

            obs = next_obs
            all_obs.append(obs.copy())

        values = agent.policy.critic_forward(all_obs).detach()

        final_report["value_mean"][i] = values.mean()
        final_report["value_std"][i] = values.std()
        final_report["value_min"][i] = values.min()
        final_report["value_max"][i] = values.max()

        all_obs.clear()

    final_report["value_mean"] = final_report["value_mean"].mean()
    final_report["value_std"] = final_report["value_std"].mean()
    final_report["value_min"] = final_report["value_min"].mean()
    final_report["value_max"] = final_report["value_max"].mean()

    print_report(final_report)

    return final_report


def actor_only_eval(
    env_fn: Callable[[], Env],
    agent: Agent,
    n_episodes: int = 5
) -> Dict[str, torch.Tensor]:
    env = env_fn()


    final_report = {
        "action_mean": torch.zeros(size=(n_episodes, agent.policy.n_actions)),
        "action_std": torch.zeros(size=(n_episodes, agent.policy.n_actions)),
        "action_min": torch.zeros(size=(n_episodes, agent.policy.n_actions)),
        "action_max": torch.zeros(size=(n_episodes, agent.policy.n_actions)),
    }
    for i in range(n_episodes):
        obs, _ = env.reset()

        terminated, truncated = False, False

        all_actions = []

        while not terminated and not truncated:
            actions, _, _ = agent.act(obs, False)
            actions = actions.to(dtype=torch.float)

            next_obs, _, terminated, truncated, _ = env.step(actions.squeeze())

            obs = next_obs
            all_actions.append(actions)

        all_actions = torch.stack(all_actions)

        final_report["action_mean"][i] = all_actions.mean(dim=1).mean(dim=0)
        final_report["action_std"][i] = all_actions.std(dim=1).mean(dim=0)
        final_report["action_min"][i] = all_actions.min()
        final_report["action_max"][i] = all_actions.max()

        del all_actions

    final_report["action_mean"] = final_report["action_mean"].mean(dim=0)
    final_report["action_std"] = final_report["action_std"].mean(dim=0)
    final_report["action_min"] = final_report["action_min"].mean(dim=0)
    final_report["action_max"] = final_report["action_max"].mean(dim=0)

    print_report(final_report)

    return final_report