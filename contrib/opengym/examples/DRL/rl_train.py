import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import os
import sys

try:
    import ns3gym
    from ns3gym import ns3env
except ImportError:
    print("Warning: ns3gym not found. Please install ns3-gym first.")
    print("Installation: pip install ns3gym")
    sys.exit(1)

# class A3CActor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128):
#         super(A3CActor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc_mean = nn.Linear(hidden_dim, action_dim)
#         self.fc_std = nn.Linear(hidden_dim, action_dim)
#
#         nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
#         nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
#         nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
#         nn.init.orthogonal_(self.fc_std.weight, gain=0.01)
#
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         mean = self.fc_mean(x)
#         std = torch.clamp(torch.exp(self.fc_std(x)), min=1e-6, max=1.0)
#         return mean, std
#
#
# class A3CCritic(nn.Module):
#     def __init__(self, state_dim, hidden_dim=128):
#         super(A3CCritic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)
#
#         nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
#         nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
#         nn.init.orthogonal_(self.fc3.weight, gain=1.0)
#
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         value = self.fc3(x)
#         return value
#
#
#
# class PPOAgent:
#     def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3,
#                  gamma=0.99, n_steps=20, entropy_coef=0.01, value_coef=0.5):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.gamma = gamma
#         self.n_steps = n_steps  # n-step return
#         self.entropy_coef = entropy_coef
#         self.value_coef = value_coef
#
#         self.action_low = np.array([0.5, 0.5, 0.01, 2.1, 2.1, 6.0])
#         self.action_high = np.array([1.5, 1.5, 0.99, 4.0, 4.0, 10.0])
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         self.actor = A3CActor(state_dim, action_dim).to(self.device)
#         self.critic = A3CCritic(state_dim).to(self.device)
#
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
#
#         # Episode buffer for n-step returns
#         self.episode_buffer = {
#             'states': [],
#             'actions': [],
#             'rewards': [],
#             'dones': [],
#             'log_probs': [],
#             'values': []
#         }
#
#     def select_action(self, state):
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#
#         self.actor.eval()
#         with torch.no_grad():
#             mean, std = self.actor(state_tensor)
#             dist = Normal(mean, std)
#             action = dist.sample()
#             log_prob = dist.log_prob(action).sum(dim=1)
#             value = self.critic(state_tensor)
#
#         action_np = action.cpu().numpy().flatten()
#         # Clamp actions to valid ranges
#         action_np = np.clip(action_np, self.action_low, self.action_high)
#
#         return action_np, log_prob.item(), value.item()
#
#     def store_transition(self, state, action, reward, done, log_prob, value):
#         self.episode_buffer['states'].append(state)
#         self.episode_buffer['actions'].append(action)
#         self.episode_buffer['rewards'].append(reward)
#         self.episode_buffer['dones'].append(done)
#         self.episode_buffer['log_probs'].append(log_prob)
#         self.episode_buffer['values'].append(value)
#
#     def compute_n_step_returns(self, next_value=0.0):
#         rewards = self.episode_buffer['rewards']
#         values = self.episode_buffer['values']
#         dones = self.episode_buffer['dones']
#
#         returns = []
#         advantages = []
#
#         # Compute n-step returns
#         R = next_value
#         for t in reversed(range(len(rewards))):
#             if dones[t]:
#                 R = 0.0
#             R = rewards[t] + self.gamma * R
#             returns.insert(0, R)
#
#         for i in range(len(returns)):
#             advantage = returns[i] - values[i]
#             advantages.append(advantage)
#
#         return returns, advantages
#
#     def update(self, next_value=0.0):
#         if len(self.episode_buffer['states']) == 0:
#             return
#
#         returns, advantages = self.compute_n_step_returns(next_value)
#
#         states = torch.FloatTensor(self.episode_buffer['states']).to(self.device)
#         actions = torch.FloatTensor(self.episode_buffer['actions']).to(self.device)
#         returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
#         advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
#         old_log_probs = torch.FloatTensor(self.episode_buffer['log_probs']).unsqueeze(1).to(self.device)
#
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
#
#         mean, std = self.actor(states)
#         dist = Normal(mean, std)
#         new_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
#         entropy = dist.entropy().sum(dim=1, keepdim=True)
#
#         actor_loss = -(new_log_probs * advantages).mean() - self.entropy_coef * entropy.mean()
#
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
#         self.actor_optimizer.step()
#
#         values = self.critic(states)
#         critic_loss = self.value_coef * nn.MSELoss()(values, returns)
#
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
#         self.critic_optimizer.step()
#
#         self.episode_buffer = {
#             'states': [],
#             'actions': [],
#             'rewards': [],
#             'dones': [],
#             'log_probs': [],
#             'values': []
#         }
#
#     def save(self, filepath):
#         torch.save({
#             'actor': self.actor.state_dict(),
#             'critic': self.critic.state_dict(),
#             'actor_optimizer': self.actor_optimizer.state_dict(),
#             'critic_optimizer': self.critic_optimizer.state_dict(),
#         }, filepath)
#         print(f"Model saved to {filepath}")
#
#     def load(self, filepath):
#         checkpoint = torch.load(filepath, map_location=self.device)
#         self.actor.load_state_dict(checkpoint['actor'])
#         self.critic.load_state_dict(checkpoint['critic'])
#         if 'actor_optimizer' in checkpoint:
#             self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
#         if 'critic_optimizer' in checkpoint:
#             self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
#         print(f"Model loaded from {filepath}")



# class DDPGActor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(DDPGActor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#         
#         nn.init.uniform_(self.fc1.weight, -1/np.sqrt(state_dim), 1/np.sqrt(state_dim))
#         nn.init.uniform_(self.fc2.weight, -1/np.sqrt(hidden_dim), 1/np.sqrt(hidden_dim))
#         nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
#         
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         # Output actions directly (deterministic policy)
#         action = torch.tanh(self.fc3(x))
#         return action
# 
# 
# class DDPGCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(DDPGCritic, self).__init__()
#         self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)
#         
#         nn.init.uniform_(self.fc1.weight, -1/np.sqrt(state_dim + action_dim), 1/np.sqrt(state_dim + action_dim))
#         nn.init.uniform_(self.fc2.weight, -1/np.sqrt(hidden_dim), 1/np.sqrt(hidden_dim))
#         nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
#         
#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         q_value = self.fc3(x)
#         return q_value
# 
# 
# class ReplayBuffer:
#     def __init__(self, capacity=100000):
#         self.buffer = deque(maxlen=capacity)
#     
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#     
#     def sample(self, batch_size):
#         import random
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         
#         return (np.array(states),
#                 np.array(actions),
#                 np.array(rewards, dtype=np.float32),
#                 np.array(next_states),
#                 np.array(dones, dtype=np.float32))
#     
#     def __len__(self):
#         return len(self.buffer)
# 
# 
# class DDPGAgent:
#     def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3,
#                  gamma=0.99, tau=0.001, noise_std=0.1, noise_decay=0.995,
#                  min_noise=0.05, buffer_size=100000, batch_size=64):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.gamma = gamma
#         self.tau = tau  # Soft update coefficient for target networks
#         self.noise_std = noise_std
#         self.noise_decay = noise_decay
#         self.min_noise = min_noise
#         self.batch_size = batch_size
#         
#         self.action_low = np.array([0.5, 0.5, 0.01, 2.1, 2.1, 6.0])
#         self.action_high = np.array([1.5, 1.5, 0.99, 4.0, 4.0, 10.0])
#         
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         
#         self.actor = DDPGActor(state_dim, action_dim).to(self.device)
#         self.actor_target = DDPGActor(state_dim, action_dim).to(self.device)
#         self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
#         self.critic_target = DDPGCritic(state_dim, action_dim).to(self.device)
#         
#         self.actor_target.load_state_dict(self.actor.state_dict())
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
#         
#         self.replay_buffer = ReplayBuffer(buffer_size)
#         
#     def select_action(self, state, add_noise=True):
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         self.actor.eval()
#         with torch.no_grad():
#             action = self.actor(state).cpu().numpy().flatten()
#         self.actor.train()
#         
#         if add_noise:
#             noise = np.random.normal(0, self.noise_std, size=action.shape)
#             action = action + noise
#         
#         action = np.clip(action, self.action_low, self.action_high)
#         
#         return action
#     
#     def store_transition(self, state, action, reward, next_state, done):
#         self.replay_buffer.push(state, action, reward, next_state, done)
#     
#     def update(self):
#         if len(self.replay_buffer) < self.batch_size:
#             return
#         
#         states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
#         
#         states = torch.FloatTensor(states).to(self.device)
#         actions = torch.FloatTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
#         
#         with torch.no_grad():
#             next_actions = self.actor_target(next_states)
#             next_q_values = self.critic_target(next_states, next_actions)
#             target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
#         
#         current_q_values = self.critic(states, actions)
#         critic_loss = nn.MSELoss()(current_q_values, target_q_values)
#         
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
#         self.critic_optimizer.step()
#         
#         predicted_actions = self.actor(states)
#         actor_loss = -self.critic(states, predicted_actions).mean()
#         
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
#         self.actor_optimizer.step()
#         
#         self._soft_update(self.actor_target, self.actor, self.tau)
#         self._soft_update(self.critic_target, self.critic, self.tau)
#         
#         self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)
#     
#     def _soft_update(self, target, source, tau):
#         for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
#     
#     def save(self, filepath):
#         torch.save({
#             'actor': self.actor.state_dict(),
#             'actor_target': self.actor_target.state_dict(),
#             'critic': self.critic.state_dict(),
#             'critic_target': self.critic_target.state_dict(),
#             'actor_optimizer': self.actor_optimizer.state_dict(),
#             'critic_optimizer': self.critic_optimizer.state_dict(),
#         }, filepath)
#         print(f"Model saved to {filepath}")
#     
#     def load(self, filepath):
#         checkpoint = torch.load(filepath, map_location=self.device)
#         self.actor.load_state_dict(checkpoint['actor'])
#         self.actor_target.load_state_dict(checkpoint.get('actor_target', checkpoint['actor']))
#         self.critic.load_state_dict(checkpoint['critic'])
#         self.critic_target.load_state_dict(checkpoint.get('critic_target', checkpoint['critic']))
#         print(f"Model loaded from {filepath}")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.fc_std.weight, gain=0.01)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.clamp(torch.exp(self.fc_std(x)), min=1e-6, max=1.0)
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4,
                 gamma=0.95, eps_clip=0.2, k_epochs=10, gae_lambda=0.95,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, batch_size=64):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_old = Actor(state_dim, action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'is_terminals': [],
            'logprobs': [],
            'next_states': []
        }

    def select_action(self, state):
        from torch.distributions import Normal
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor_old(state)
            dist = Normal(mean, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=1)
        action = torch.clamp(action,
                            torch.tensor([0.5, 0.5, 0.01, 2.1, 2.1, 6.0]),
                            torch.tensor([1.5, 1.5, 0.99, 4.0, 4.0, 10.0]))
        return action.squeeze().numpy(), action_logprob.item()

    def store_transition(self, state, action, reward, is_terminal, logprob, next_state=None):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['is_terminals'].append(is_terminal)
        self.memory['logprobs'].append(logprob)
        self.memory['next_states'].append(next_state if next_state is not None else state)

    def update(self):
        if len(self.memory['states']) == 0:
            return
        from torch.distributions import Normal
        old_states = torch.FloatTensor(self.memory['states'])
        old_actions = torch.FloatTensor(self.memory['actions'])
        old_logprobs = torch.FloatTensor(self.memory['logprobs'])
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory['rewards']),
                                      reversed(self.memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.FloatTensor(rewards)
        with torch.no_grad():
            values = self.critic(old_states).squeeze()
            next_states = torch.FloatTensor(self.memory['next_states'])
            next_values = self.critic(next_states).squeeze()
            is_terminals_tensor = torch.BoolTensor(self.memory['is_terminals'])
            next_values[is_terminals_tensor] = 0.0
            td_errors = rewards + self.gamma * next_values - values
            advantages = torch.zeros_like(td_errors)
            gae = 0
            for t in reversed(range(len(td_errors))):
                if is_terminals_tensor[t]:
                    gae = 0
                gae = td_errors[t] + self.gamma * self.gae_lambda * gae
                advantages[t] = gae
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + values
        total_samples = old_states.size(0)
        for epoch in range(self.k_epochs):
            indices = torch.randperm(total_samples)
            for i in range(0, total_samples, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                if len(batch_indices) == 0:
                    continue
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_returns = returns[batch_indices]
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                batch_new_logprobs = dist.log_prob(batch_actions).sum(dim=1)
                batch_entropy = dist.entropy().sum(dim=1)
                batch_ratio = torch.exp(batch_new_logprobs - batch_old_logprobs.squeeze())
                batch_surr1 = batch_ratio * batch_advantages.squeeze()
                batch_surr2 = torch.clamp(batch_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages.squeeze()
                batch_actor_loss = -torch.min(batch_surr1, batch_surr2).mean() - self.entropy_coef * batch_entropy.mean()
                batch_values = self.critic(batch_states).squeeze()
                batch_critic_loss = self.value_coef * nn.MSELoss()(batch_values, batch_returns.squeeze())
                self.actor_optimizer.zero_grad()
                batch_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                batch_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'is_terminals': [],
            'logprobs': [],
            'next_states': []
        }

    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_old.load_state_dict(self.actor.state_dict())
        print(f"Model loaded from {filepath}")


def train_agent(port=5555, max_episodes=1000, max_steps=1000, total_timesteps=5000000):
    env = ns3env.Ns3Env(port=port)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Total timesteps: {total_timesteps}")
    agent = PPOAgent(state_dim, action_dim)
    episode_rewards = []
    episode_lengths = []
    timestep_count = 0
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, done, log_prob, next_state)
            state = next_state
            episode_reward += reward
            episode_length += 1
            timestep_count += 1
            if done:
                break
        agent.update()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode+1}/{max_episodes}, "
                  f"Timesteps: {timestep_count}/{total_timesteps}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.2f}")
        if (episode + 1) % 100 == 0:
            agent.save(f"fusion_rl_model_ep{episode+1}.pth")
        if timestep_count >= total_timesteps:
            print(f"Reached total timesteps limit: {total_timesteps}")
            break
    agent.save("fusion_rl_model_final.pth")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train RL agent for Fusion')
    parser.add_argument('--port', type=int, default=5555, help='NS-3 gym port')
    parser.add_argument('--episodes', type=int, default=1000, help='Max episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--total-timesteps', type=int, default=5000000, help='Total training timesteps')
    args = parser.parse_args()
    
    train_agent(port=args.port, max_episodes=args.episodes, max_steps=args.steps, 
                total_timesteps=args.total_timesteps)
