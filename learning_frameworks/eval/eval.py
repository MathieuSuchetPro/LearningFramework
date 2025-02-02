from typing import Dict, Callable

import numpy as np
import torch
from gymnasium import Env

from learning_frameworks.learning.PPO import PPO


def run_eval(
        env_fn: Callable[[], Env],
        agent: PPO,

        n_episodes: int = 5,
        _print_report: bool = False
) -> Dict[str, float]:
    env = env_fn()

    report = {
        "Average reward per step": np.zeros(shape=(n_episodes, 1)),
        "Average episode length": np.zeros(shape=(n_episodes, 1)),
        "Average reward per episode": np.zeros(shape=(n_episodes, 1))
    }

    for i in range(n_episodes):
        obs, _ = env.reset()

        terminated, truncated = False, False

        all_r = []
        ep_len = 0

        while not truncated and not terminated:
            actions, _, _ = agent.act(torch.tensor(obs, dtype=torch.float), deterministic=False)
            actions = actions.numpy()

            next_obs, r, terminated, truncated, _ = env.step(actions)

            if truncated:
                r = agent.value_estimator(next_obs).detach().numpy()


            all_r.append(r)

            obs = next_obs

            ep_len += 1

        report["Average reward per step"][i] = sum(all_r) / len(all_r)
        report["Average reward per episode"][i] = sum(all_r)
        report["Average episode length"][i] = ep_len

    report["Average reward per step"] = report["Average reward per step"].mean()
    report["Average episode length"] = report["Average episode length"].mean()
    report["Average reward per episode"] = report["Average reward per episode"].mean()

    if _print_report:
        print_report(report)

    return report


def print_report(report: Dict[str, float]):
    print(f"{' Eval report ':=^40}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k: <40}: {v:.3f}")
        else:
            print(f"{k: <40}: {v}")
    print(f"{' End of report ':=^40}")
