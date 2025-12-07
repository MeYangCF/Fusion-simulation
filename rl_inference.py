#!/usr/bin/env python3

import gym
import numpy as np
import torch
import torch.nn as nn
import sys

try:
    import ns3gym
    from ns3gym import ns3env
except ImportError:
    print("Warning: ns3gym not found. Please install ns3-gym first.")
    sys.exit(1)

from rl_train import PPOAgent


# def run_inference(model_path, port=5555, max_episodes=10, max_steps=1000):
#     env = ns3env.Ns3Env(port=port)
#
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#
#     agent = RLAgent(state_dim, action_dim)
#     agent.load(model_path)
#     agent.actor.eval()  # Set to evaluation mode
#
#     print(f"Loaded model from {model_path}")
#     print(f"Running inference on port {port}")
#
#     for episode in range(max_episodes):
#         state = env.reset()
#         episode_reward = 0
#         episode_length = 0
#
#         for step in range(max_steps):
#             # Select action (deterministic: use mean of policy distribution)
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
#             with torch.no_grad():
#                 mean, std = agent.actor(state_tensor)
#                 action = mean.cpu().numpy().flatten()
#
#             action = np.clip(action, agent.action_low, agent.action_high)
#
#
#             next_state, reward, done, info = env.step(action)
#
#             state = next_state
#             episode_reward += reward
#             episode_length += 1
#
#             if done:
#                 break
#
#         print(f"Episode {episode+1}/{max_episodes}, "
#               f"Reward: {episode_reward:.2f}, "
#               f"Length: {episode_length}")
#
#     env.close()



# def run_inference_ddpg(model_path, port=5555, max_episodes=10, max_steps=1000):
#     env = ns3env.Ns3Env(port=port)
#     
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     
#     agent = RLAgent(state_dim, action_dim)
#     agent.load(model_path)
#     agent.actor.eval()  # Set to evaluation mode
#     
#     print(f"Loaded model from {model_path}")
#     print(f"Running inference on port {port}")
#     
#     for episode in range(max_episodes):
#         state = env.reset()
#         episode_reward = 0
#         episode_length = 0
#         
#         for step in range(max_steps):
#             # Select action (deterministic, no exploration noise)
#             action = agent.select_action(state, add_noise=False)
#             
#             # Execute action
#             next_state, reward, done, info = env.step(action)
#             
#             state = next_state
#             episode_reward += reward
#             episode_length += 1
#             
#             if done:
#                 break
#         
#         print(f"Episode {episode+1}/{max_episodes}, "
#               f"Reward: {episode_reward:.2f}, "
#               f"Length: {episode_length}")
#     
#     env.close()


def run_inference_ppo(model_path, port=5555, max_episodes=10, max_steps=1000):
    env = ns3env.Ns3Env(port=port)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)
    agent.load(model_path)
    agent.actor.eval()
    print(f"Loaded model from {model_path}")
    print(f"Running inference on port {port}")
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        for step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                mean, std = agent.actor(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.mean
                action = torch.clamp(action,
                                   torch.tensor([0.5, 0.5, 0.01, 2.1, 2.1, 6.0]),
                                   torch.tensor([1.5, 1.5, 0.99, 4.0, 4.0, 10.0]))
                action = action.squeeze().numpy()
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            episode_length += 1
            if done:
                break
        print(f"Episode {episode+1}/{max_episodes}, "
              f"Reward: {episode_reward:.2f}, "
              f"Length: {episode_length}")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run RL inference for Fusion')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--port', type=int, default=5555, help='NS-3 gym port')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Max steps per episode')
    args = parser.parse_args()
    
    run_inference_ppo(args.model, port=args.port, 
                  max_episodes=args.episodes, max_steps=args.steps)
