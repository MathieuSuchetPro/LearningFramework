from typing import Callable

from gymnasium import Env

from learning_frameworks.learning.PPO import PPO


def render_loop(agent: PPO, env_fn: Callable[[], Env]):
    env = env_fn()
    while True:
        obs, _ = env.reset()

        terminated, truncated = False, False

        while not terminated and not truncated:

            env.render()

            actions, _, _ = agent.act(obs, deterministic=False)
            actions = actions.numpy()

            next_obs, _, terminated, truncated, _ = env.step(actions)

            obs = next_obs