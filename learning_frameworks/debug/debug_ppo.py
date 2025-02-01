import torch

from learning_frameworks.learning.PPO import PPO
from learning_frameworks.collection.buffer import Buffer
from learning_frameworks.collection.collection import Collection
from learning_frameworks.eval.debug_eval import critic_only_eval, actor_only_eval
from learning_frameworks.policies.continuous_policy import ContinuousPolicy
from learning_frameworks.debug.probe_envs.probe_envs import MultiDiscreteEnv

if __name__ == "__main__":
    def create_env():
        return MultiDiscreteEnv()


    bins = [5, 3, 5, 2]

    input_shape = 1
    output_shape = sum(bins)

    buffer_size = 1_000

    run_name = "debug_critic_isolation_5"


    policy = ContinuousPolicy(
        input_size=input_shape,
        output_size=2,

        actor_layer_sizes=[64, 64],
        critic_layer_sizes=[128, 128],

        actor_lr=1e-4,
        critic_lr=1e-4,
    )

    agent = PPO(
        policy=policy,
        device=torch.device("cpu"),

        ppo_batch_size=buffer_size // 5,
        ppo_minibatch_size=buffer_size // 20,

        policy_max_grad_norm=0.5,
        critic_max_grad_norm=0.5,

        ent_coef=0.05,
        critic_loss_coef=1,

        gae_gamma=0.99,
        gae_lambda=0.95,

        ppo_policy_clip=0.1,
        ppo_epochs=10,

        run_name=run_name,

        save_every_n=2
    )

    try:
        print(f"Trying to load {'models/' + run_name}")
        agent.load("TenthTry/models/" + run_name)
        print(f"Loaded model {'models/' + run_name} successfully")
    except FileNotFoundError as e:
        print(e)
        print("An error occurred doing agent loading, ignoring")

    n_proc = 5

    buffer = Buffer(buffer_size, input_shape, policy.n_actions)

    n_iterations = 200

    collection = Collection(create_env, agent, None, buffer, n_proc)

    for _ in range(n_iterations):
        try:
            collection.collect()

            agent.learn(buffer)

            print("Running eval...")
            actor_only_eval(create_env, agent)
            critic_only_eval(create_env, agent)
            print("End of eval")

            buffer.clear()

        except KeyboardInterrupt:
            agent.save("models/" + run_name)


