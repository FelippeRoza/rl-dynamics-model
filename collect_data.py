import continuousSafetyGym
import gymnasium as gym
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # For the progress bar
import torch
import os
import argparse
from stable_baselines3 import PPO, SAC
from stable_baselines3.sac.policies import SACPolicy

def run_env(env_name, num_steps, rl_path):
    env = gym.make(env_name)
    if rl_path:
        rl_agent = SAC.load(rl_path, env=env)
    observation, info = env.reset()
    cost = info['cost']
    observations = []
    actions = []
    costs = []
    next_costs = []
    
    for _ in tqdm(range(num_steps)):
        if rl_path:
            action, _states = rl_agent.predict(observation, deterministic=True)
        else:
            action = env.action_space.sample()  # Random action        
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_cost = info['cost']
        observations.append(observation)
        actions.append(action)
        costs.append(cost)
        next_costs.append(next_cost)

        cost = next_cost
        observation = next_observation

        if terminated or truncated:
            observation, info = env.reset()
            cost = info['cost']

    env.close()
    
    # Return observations and actions as numpy arrays
    return np.array(observations), np.array(actions), np.array(costs), np.array(next_costs)

def collect_data_parallel(env_name, num_steps_per_env, num_envs, rl_path=''):
    observations = []
    actions = []
    costs = []
    next_costs = []
    
    executor = ProcessPoolExecutor()
    try:
        futures = [executor.submit(run_env, env_name, num_steps_per_env, rl_path) for _ in range(num_envs)]
        
        for future in tqdm(as_completed(futures), total=num_envs, desc="Collecting data"):
            obs, acts, cost, next_cost = future.result()
            observations.append(obs)
            actions.append(acts)
            costs.append(cost)
            next_costs.append(next_cost)
    finally:
        # Make sure the executor shuts down properly
        executor.shutdown(wait=True)
    
    observations = np.vstack(observations)
    actions = np.vstack(actions)
    costs = np.vstack(costs)
    next_costs = np.vstack(next_costs)
    
    return observations, actions, costs, next_costs

# Save the dataset as PyTorch tensors
def save_dataset(observations, actions, costs, next_costs, filename):
    dataset = {
        'observations': torch.tensor(observations, dtype=torch.float32),
        'actions': torch.tensor(actions, dtype=torch.float32),
        'costs': torch.tensor(costs, dtype=torch.float32),
        'next_costs': torch.tensor(next_costs, dtype=torch.float32),
    }
    torch.save(dataset, filename)
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collects data from continuous-safety-gym and save as a torch dataset")
    parser.add_argument('--env', type=str, help='Environment name')
    parser.add_argument('--dataset_dir', type=str, help='Dataset saving location')
    parser.add_argument('--n_steps', type=int, help='Number of total collected steps', default=500_000)
    parser.add_argument('--n_proc', type=int, help='Number of parallel processes to speed up collection', default=8)
    parser.add_argument('--rl_agent_path', type=str, help='Path to load RL agent. If none random actions will be used.', default='')
    args = parser.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)
    num_steps_per_env = int(args.n_steps/args.n_proc)  # Number of steps per environment

    observations, actions, costs, next_costs = collect_data_parallel(args.env, num_steps_per_env, args.n_proc, args.rl_agent_path)
    save_dataset(observations, actions, costs, next_costs, os.path.join(args.dataset_dir, f'{args.env}_dataset.pt'))
