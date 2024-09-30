import continuousSafetyGym
import gymnasium as gym
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # For the progress bar
import torch

def run_env(env_name, num_steps):
    env = gym.make(env_name)
    observation, info = env.reset()
    cost = info['cost']
    observations = []
    actions = []
    costs = []
    next_costs = []
    
    for _ in tqdm(range(num_steps)):
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

def collect_data_parallel(env_name, num_steps_per_env, num_envs):
    observations = []
    actions = []
    costs = []
    next_costs = []
    
    executor = ProcessPoolExecutor()
    try:
        futures = [executor.submit(run_env, env_name, num_steps_per_env) for _ in range(num_envs)]
        
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
    env_name = 'MultiagentDescentralizedSafe-v0'
    total_steps = 1_000_000
    num_envs = 25  # Number of parallel environments
    num_steps_per_env = int(total_steps/num_envs)  # Number of steps per environment

    observations, actions, costs, next_costs = collect_data_parallel(env_name, num_steps_per_env, num_envs)
    save_dataset(observations, actions, costs, next_costs, f'{env_name}_dataset.pt')
