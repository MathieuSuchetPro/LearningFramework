import multiprocessing as mp

from gymnasium import Env


def __init_metrics():
    return {
        "mean_episode_len": 0
    }


def _process(env: Env, to_parent: mp.Pipe):
    closed = False
    metrics = __init_metrics()

    n_episodes = 0
    len_episode = 0

    while not closed:
        command = to_parent.recv()

        if command["name"] == "reset":
            obs, info = env.reset()
            to_parent.send((obs, info))

        elif command["name"] == "step":

            next_obs, r, terminal, truncated, info = env.step(command["actions"])
            len_episode += 1

            reset_next_obs = next_obs
            reset_info = info

            if terminal or truncated:
                reset_next_obs, reset_info = env.reset()
                metrics["mean_episode_len"] += len_episode

                len_episode = 0
                n_episodes += 1

            to_parent.send((next_obs, r, terminal, truncated, info, reset_next_obs, reset_info))

        elif command["name"] == "close":
            closed = True
            to_parent.close()
            continue

        elif command["name"] == "get_metrics":
            if n_episodes == 0:
                n_episodes = 1
            metrics["mean_episode_len"] //= n_episodes
            to_parent.send(metrics)

            metrics = __init_metrics()
            n_episodes = 0
            len_episode = 0
