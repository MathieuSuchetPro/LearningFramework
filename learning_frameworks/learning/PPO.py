import pathlib
import warnings
from typing import Union, Dict

import numpy as np
import torch.nn
from torch.utils.tensorboard import SummaryWriter

from learning_frameworks.collection.buffer import Buffer
from learning_frameworks.learning.agent import Agent
from learning_frameworks.policies.policy import Policy
from learning_frameworks.value_estimators.value_estimator import ValueEstimator


class PPO(Agent):
    def __init__(
            self,
            policy: Policy,
            value_estimator: ValueEstimator,

            policy_max_grad_norm: float,
            critic_max_grad_norm: float,

            ent_coef: float,
            critic_loss_coef: float,

            gae_lambda: float,
            gae_gamma: float,

            ppo_batch_size: int,
            ppo_minibatch_size: int,
            ppo_policy_clip: float,
            ppo_epochs: int,

            device: torch.device,
            save_every_n: int, run_name: str):
        super().__init__(policy, value_estimator)

        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gamma

        self.ppo_batch_size = ppo_batch_size
        self.ppo_minibatch_size = ppo_minibatch_size

        if self.ppo_batch_size % self.ppo_minibatch_size:
            warnings.warn(f"Be careful, you used a batch size of {self.ppo_batch_size} with a"
                          f"minibatch size of {self.ppo_minibatch_size}, that means the last batch will have"
                          f"a size of {self.ppo_minibatch_size - (self.ppo_batch_size % self.ppo_minibatch_size)}")

        self.ppo_policy_clip = ppo_policy_clip
        self.ppo_epochs = ppo_epochs

        self.device = device
        self.run_name = run_name
        self.writer = SummaryWriter("logs/" + run_name)

        self.save_every_n = save_every_n
        self._cnt_every_n = 0

        self.ent_coef = ent_coef
        self.critic_loss_coef = critic_loss_coef

        self.policy_max_grad_norm = policy_max_grad_norm
        self.critic_max_grad_norm = critic_max_grad_norm

    def compute_advantages(self, values, rewards, dones):
        advantages = torch.zeros(size=(1, len(values))).to(self.device)
        last_advantage = 0

        last_value = values[-1]

        for t in reversed(range(len(values))):
            mask = 1.0 - dones[t]
            last_value = last_value * mask

            last_advantage = last_advantage * mask
            delta = rewards[t] + self.gae_gamma * last_value - values[t]

            last_advantage = delta + self.gae_gamma * self.gae_lambda * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[t]

        return torch.squeeze(advantages)

    def learn(self, buffer: Buffer):

        n_iterations = 0
        n_minibatch_iterations = 0
        mean_entropy = 0
        mean_policy_loss = 0
        mean_val_loss = 0
        mean_reward = 0
        mean_divergence = 0

        # Advantages
        mean_adv_std = 0
        mean_adv_max = 0
        mean_adv = 0
        mean_adv_min = 0

        # Save parameters before computing any updates.
        policy_before = torch.nn.utils.parameters_to_vector(
            self.policy.nn.parameters()
        ).cpu()
        critic_before = torch.nn.utils.parameters_to_vector(
            self.value_estimator.nn.parameters()
        ).cpu()

        for epoch in range(self.ppo_epochs):

            # Get every batches
            states_arr, actions_arr, rewards_arr, dones_arr, values_arr, entropies_arr, log_probs_arr, batches = buffer.get_batches(
                self.ppo_batch_size)
            advantages_arr = self.compute_advantages(values_arr, rewards_arr, dones_arr)

            minibatch_ratio = self.ppo_minibatch_size / self.ppo_batch_size

            for batch in batches:
                batch_states = states_arr[batch].to(self.device)
                batch_actions = actions_arr[batch].to(self.device)
                batch_rewards = rewards_arr[batch].to(self.device)

                batch_log_probs = log_probs_arr[batch].to(self.device)
                batch_advantages = advantages_arr[batch].to(self.device)

                batch_values = values_arr[batch].to(self.device)
                batch_entropies = entropies_arr[batch].to(self.device)

                self.policy.optimizer.zero_grad()
                self.value_estimator.optimizer.zero_grad()

                for minibatch in np.arange(start=0, stop=self.ppo_batch_size, step=self.ppo_minibatch_size):
                    start = minibatch
                    stop = start + self.ppo_minibatch_size

                    # Calculate ratio
                    states = batch_states[start:stop]
                    log_probs = batch_log_probs[start:stop].squeeze()
                    actions = batch_actions[start:stop].squeeze()

                    new_entropies, new_log_probs = self.policy.get_backprop_data(states, actions)
                    new_values = self.value_estimator.get_backprop_data(states, None)

                    prob_ratio = torch.exp(new_log_probs - log_probs)

                    # Normalized advantages
                    advantages = batch_advantages[start:stop].squeeze()
                    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                    mean_adv_std += advantages.std()
                    mean_adv_min += advantages.min()
                    mean_adv_max += advantages.max()
                    mean_adv += advantages.mean()

                    weighted_probs = advantages * prob_ratio
                    weighted_clipped_probs = torch.clamp(
                        prob_ratio,
                        1 - self.ppo_policy_clip,
                        1 + self.ppo_policy_clip
                    ) * advantages

                    with torch.no_grad():
                        log_ratio = log_probs - batch_log_probs
                        kl = (torch.exp(log_ratio) - 1) - log_ratio
                        kl = kl.mean().detach().cpu().item()

                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    values = batch_values[start:stop].squeeze()
                    returns = advantages + values
                    # returns = (returns - returns.mean()) / (returns.std() + 1e-10)

                    critic_loss = 0.5 * (returns - new_values) ** 2
                    critic_loss = critic_loss.mean()

                    entropies = batch_entropies[start:stop]

                    loss = actor_loss + entropies.mean() * self.ent_coef + critic_loss * self.critic_loss_coef
                    loss *= minibatch_ratio

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value_estimator.parameters(), self.critic_max_grad_norm)

                    rewards = batch_rewards[start:stop]

                    mean_entropy += entropies.mean()
                    mean_divergence += kl
                    mean_reward += rewards.mean()

                    mean_policy_loss += actor_loss.mean().cpu().detach().item() / minibatch_ratio
                    mean_val_loss += critic_loss.mean().cpu().detach().item() / minibatch_ratio

                    n_minibatch_iterations += 1

                self.policy.optimizer.step()
                self.value_estimator.optimizer.step()

                n_iterations += 1

        if n_iterations == 0:
            n_iterations = 1

        if n_minibatch_iterations == 0:
            n_minibatch_iterations = 1

        mean_entropy /= n_minibatch_iterations
        mean_reward /= n_minibatch_iterations

        mean_policy_loss /= n_minibatch_iterations
        mean_val_loss /= n_minibatch_iterations
        mean_divergence /= n_minibatch_iterations

        mean_adv /= n_minibatch_iterations
        mean_adv_std /= n_minibatch_iterations
        mean_adv_min /= n_minibatch_iterations
        mean_adv_max /= n_minibatch_iterations

        policy_after = torch.nn.utils.parameters_to_vector(
            self.policy.nn.parameters()
        ).cpu()
        critic_after = torch.nn.utils.parameters_to_vector(
            self.value_estimator.nn.parameters()
        ).cpu()

        # Metrics
        policy_update_magnitude = (policy_before - policy_after).norm().item()
        critic_update_magnitude = (critic_before - critic_after).norm().item()

        self.writer.add_scalar("Loss/actor", mean_policy_loss, self.policy.learning_step)
        self.writer.add_scalar("Loss/critic", mean_val_loss, self.policy.learning_step)

        self.writer.add_scalar("Learning_metrics/entropy", mean_entropy, self.policy.learning_step)
        self.writer.add_scalar("Learning_metrics/rewards", mean_reward, self.policy.learning_step)

        self.writer.add_scalar("Policy/actor_update_magnitude", policy_update_magnitude, self.policy.learning_step)
        self.writer.add_scalar("Policy/critic_update_magnitude", critic_update_magnitude, self.policy.learning_step)
        self.writer.add_scalar("Policy/mean_kl_divergence", mean_divergence, self.policy.learning_step)

        self.writer.add_scalar("Critic/mean_advantage", mean_adv, self.policy.learning_step)
        self.writer.add_scalar("Critic/mean_advantage_std", mean_adv_std, self.policy.learning_step)
        self.writer.add_scalar("Critic/mean_advantage_min", mean_adv_min, self.policy.learning_step)
        self.writer.add_scalar("Critic/mean_advantage_max", mean_adv_max, self.policy.learning_step)

        self.policy.learning_step += 1
        self.value_estimator.learning_rate += 1
        
        self._cnt_every_n += 1

        if self._cnt_every_n >= self.save_every_n:
            self.save("models/" + self.run_name)
            self._cnt_every_n = 0

    def update_env_logs(self, logs: Dict[str, Union[float, int]]):
        for k, v in logs.items():
            self.writer.add_scalar("env/" + k, v, self.policy.learning_step)

    def save(self, path: str):
        path_ = pathlib.Path(path)

        self.policy.save(path_ / "policy")
        self.value_estimator.save(path_ / "value_estimator")

    def load(self, path: str):
        path_ = pathlib.Path(path)

        self.policy.load(path_ / "policy")
        self.value_estimator.load(path_ / "value_estimator")

    def __del__(self):
        self.writer.close()
