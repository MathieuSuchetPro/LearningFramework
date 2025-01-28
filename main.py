import torch
from gymnasium import make

from agent import Agent
from collection.buffer import Buffer
from collection.collection import Collection
from eval.eval import run_eval
from policies.continuous_policy import ContinuousPolicy

if __name__ == "__main__":

    env_id = "LunarLander-v2"

    def create_env():
        return make(env_id, continuous=True)

    env = create_env()

    input_shape = env.observation_space.shape[0]
    output_shape = env.action_space.shape[0]

    buffer_size = 10_000

    run_name = env_id + "_test_continuous_10k_batch_no_ent_loss"


    policy = ContinuousPolicy(
        input_size=input_shape,
        output_size=2,

        actor_layer_sizes=[64, 64],
        critic_layer_sizes=[128, 128],

        actor_lr=1e-4,
        critic_lr=1e-4,
    )

    agent = Agent(
        policy=policy,
        device=torch.device("cpu"),

        ppo_batch_size=buffer_size // 5,
        ppo_minibatch_size=buffer_size // 20,

        policy_max_grad_norm=0.5,
        critic_max_grad_norm=0.5,

        ent_coef=0.5,
        critic_loss_coef=1,

        gae_gamma=0.99,
        gae_lambda=0.95,

        ppo_policy_clip=0.1,
        ppo_epochs=10,

        run_name=run_name,

        save_every_n=2
    )

    try:
        print(f"Trying to load {'TenthTry/models/' + run_name}")
        agent.load("TenthTry/models/" + run_name)
        print(f"Loaded model {'TenthTry/models/' + run_name} successfully")
    except FileNotFoundError as e:
        print(e)
        print("An error occurred doing agent loading, ignoring")

    n_proc = 5

    buffer = Buffer(buffer_size, input_shape, policy.n_actions)

    n_iterations = 200

    collection = Collection(create_env, agent, None, buffer, n_proc)

    for _ in range(n_iterations):
        try:
            metrics = collection.collect()

            mean_ep_len = 0
            for metric in metrics:
                mean_ep_len += metric["mean_episode_len"]

            mean_ep_len /= n_proc

            agent.update_env_logs({
                "mean_episode_len": mean_ep_len
            })

            agent.learn(buffer)

            print("Running eval...")
            run_eval(create_env, agent, _print_report=True)
            print("End of eval")

            buffer.clear()

        except KeyboardInterrupt:
            agent.save("TenthTry/models/" + run_name)


