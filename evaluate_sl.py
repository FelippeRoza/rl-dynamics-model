from train_with_collected_data import SafetyLayerNN, GymDataset, load_model
import continuousSafetyGym
import gymnasium as gym
import numpy as np
from tqdm import tqdm  # For the progress bar
import torch
import torch.nn as nn

n_steps = 1000
env_name = 'ContSafetyBallReach-v0'
env_name = 'MultiagentDescentralizedSafe-v0'
# env_name = 'SpaceshipSafe-v0'
model_dir = 'data/sl_models/'
env = gym.make(env_name)
next_cost_list = []
next_cost_pred_list = []

obs, info = env.reset()
cost = info['cost']

model = SafetyLayerNN(input_dim=len(obs)+len(cost)+env.action_space.shape[0], 
                      output_dim=len(cost))
load_model(model, f'{model_dir}/{env_name}_sl_model.pth')
model.eval()


for _ in tqdm(range(n_steps)):
    action = env.action_space.sample()  # Random action
    
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        cost_tensor = torch.tensor(cost, dtype=torch.float32)
        input = torch.cat((obs_tensor, action_tensor, cost_tensor), dim=-1)
        next_cost_pred = model(input)
        next_cost_pred_list.append(next_cost_pred)
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    next_cost = info['cost']
    next_cost_list.append(next_cost)

    cost = next_cost
    obs = next_obs

    if terminated or truncated:
        obs, info = env.reset()  # Reset if episode finishes
        cost = info['cost']

env.close()

MSEloss = nn.MSELoss()

next_cost_pred_list = torch.stack(next_cost_pred_list)
next_cost_list = torch.tensor(next_cost_list)
loss = MSEloss(next_cost_pred_list, next_cost_list)
print(loss)

print('finished')