# coding:utf-8

import torch
import torch.nn as nn
import numpy as np

from collections import namedtuple

from models import Actor, Critic

class PPOMemory:
    memory_data = namedtuple("MemoryData", ["obs", "old_probs", "actions", "values", "rewards", "done"])

    def __init__(self):
        self.obs = []
        self.old_probs = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.done = []

        self._memory_keys = ["obs", "old_probs", "actions", "values", "rewards", "done"]
    
    def generate_mini_batches_index(self, bsz):
        total_length = len(self.obs)
        batch_num = total_length // bsz if total_length % bsz == 0 else total_length // bsz + 1

        indices = [i for i in range(total_length)]
        np.random.shuffle(indices)
        mini_batch_index = []
        for b in range(batch_num):
            mini_batch_index.append(indices[b * bsz : (b + 1) * bsz])
        
        return mini_batch_index

    def get_all_data_in_batch(self) -> memory_data:
        return self.memory_data(
            obs = self.obs,
            old_probs = self.old_probs,
            actions = self.actions,
            values = self.values,
            rewards = self.rewards,
            done = self.done
        )
    
    def save_memory(self, obs, log_prob, action, value, reward, done):
        self.obs.append(obs)
        self.old_probs.append(log_prob)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.done.append(done)

    def clear_memory(self):
        for key in self._memory_keys:
            setattr(self, key, [])

class Agent:
    def __init__(self, n_actions, input_dim, hidden_dim, gamma = 0.99, gae_lambda = 0.95, policy_clip=0.2, batch_size=64, n_epochs=10, lr=3e-4):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.actor = Actor(input_dim=input_dim, n_actions=n_actions, hidden_dim=hidden_dim, lr=lr)
        self.critic = Critic(input_dim=input_dim, hidden_dim=hidden_dim, lr=lr)
        self.memory = PPOMemory()

    def get_advantages_returns(self, values, rewards, done):
        """
            Calculate the advantages and returns.
        """

        total_length = values.size(0)
        last_gae_val = 0.0
        reversed_adv = []

        for t in reversed(range(total_length)):
            mask = 1.0 if done[t] == 0 else 0.0
            last_val = values[t + 1] if t < total_length - 1 else 0.0
            delta = rewards[t] + self.gamma * last_val * mask - values[t]
            last_gae_val = delta + self.gae_lambda * self.gamma * last_gae_val * mask
            reversed_adv.append(last_gae_val)
        
        adv = torch.tensor(reversed_adv[::-1], dtype=values.dtype, device=values.device)
        returns = adv + values

        return returns, adv

    def learn(self):
        all_data = self.memory.get_all_data_in_batch()
        values = torch.tensor(all_data.values, dtype=torch.float, device=self.actor.device)         # [_]
        rewards = torch.tensor(all_data.rewards, dtype=torch.float, device=self.actor.device)       # [_]
        obs = torch.tensor(all_data.obs, dtype=torch.float, device=self.actor.device)               # [_, h]
        old_probs = torch.tensor(all_data.old_probs, dtype=torch.float, device=self.actor.device)   # [_]
        actions = torch.tensor(all_data.actions, dtype=torch.int, device=self.actor.device)                  # [_]

        returns, advantages = self.get_advantages_returns(values=values, rewards=rewards, done=all_data.done)

        for epoch in range(self.n_epochs):
            mini_batch_index = self.memory.generate_mini_batches_index(self.batch_size)

            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0

            for mini_index in mini_batch_index:
                mini_index = torch.tensor(mini_index, dtype=torch.int, device=self.actor.device)   # [bsz]

                mini_obs = obs[mini_index]                 # [bsz, h]
                mini_old_probs = old_probs[mini_index]     # [bsz]
                mini_actions = actions[mini_index]         # [bsz]
                
                mini_returns = returns[mini_index]         # [bsz]
                mini_advs = advantages[mini_index]         # [bsz]

                new_dist = self.actor(mini_obs)            # [bsz, n]
                new_cirtic_val = self.critic(mini_obs)     # [bsz, 1]
                
                new_probs = new_dist.log_prob(mini_actions)     # [bsz]
                prob_ratio = (new_probs - mini_old_probs).exp() # [bsz]
                norm_term = prob_ratio * mini_advs              # [bsz]
                clip_term = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * mini_advs # [bsz]
                actor_loss = - torch.min(norm_term, clip_term).mean()    # [bsz]

                cirtic_loss = (mini_returns - new_cirtic_val) ** 2
                cirtic_loss = cirtic_loss.mean()

                total_loss = actor_loss + 0.5 * cirtic_loss
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += cirtic_loss.item()

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            
            # print(f"Epcoh: {epoch}, Actor loss: {epoch_actor_loss / self.batch_size:.5f}, Critic loss: {epoch_critic_loss / self.batch_size:.5f}")
        
        self.memory.clear_memory()
    
    def choose_actions(self, obs):
        obs = torch.tensor([obs], dtype=torch.float).to(self.actor.device)

        out = self.actor(obs)
        value = self.critic(obs)
        action = out.sample()

        log_prob = out.log_prob(action).squeeze().item()
        action = action.squeeze().item()
        value = value.squeeze().item()

        return log_prob, action, value

    def remember(self, obs, action, log_prob, val, reward, done):
        self.memory.save_memory(obs, log_prob, action, val, reward, done)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

if __name__ == "__main__":
    # device = "mps" if torch.backends.mps.is_available() else "cpu"

    # vals = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float, device=device)
    # rewards = torch.tensor([1.2, 2.2, 3.2, 4.2, 5.2, 6.2], dtype=torch.float, device=device)
    # done = [0, 0, 1, 0, 0, 0]

    # agent = Agent()
    # returns, adv = agent.get_advantages_returns(values=vals, rewards=rewards, done=done)
    # print(returns)
    # print(adv)

    ppo_memory = PPOMemory()
    ppo_memory.obs = [i for i in range(10)]
    indices = ppo_memory.generate_mini_batches_index(3)
    batch_data = ppo_memory.get_all_data_in_batch()
    print(batch_data)
    ppo_memory.clear_memory()
    batch_data = ppo_memory.get_all_data_in_batch()
    print(batch_data)