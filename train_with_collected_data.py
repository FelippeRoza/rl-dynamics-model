import torch
import torch.nn as nn
import torch.optim as optim
from costDynamicsModel import CDM
import continuousSafetyGym
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from costDynamicsModel.models import RegressionNN


class GymDataset(Dataset):
    def __init__(self, dataset_path):
        # Load the saved dataset from disk
        data = torch.load(dataset_path)
        self.observations = data['observations']
        self.actions = data['actions'] 
        self.costs = data['costs']
        self.next_costs = data['next_costs']

    def __len__(self):
        return len(self.observations)

    def in_dim(self):
        out_shape = torch.cat((self.observations[0], self.actions[0], self.costs[0]), dim=-1).shape
        return out_shape[0]

    def out_dim(self):
        out_shape = self.next_costs[0].shape
        return out_shape[0]

    def __getitem__(self, idx):
        obs, acts, costs= (self.observations[idx], self.actions[idx], self.costs[idx])
        input = torch.cat((obs, acts, costs), dim=-1)
        target = self.next_costs[idx]
        return input, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collects data from continuous-safety-gym and save as a torch dataset")
    parser.add_argument('--env', type=str, help='Environment name')
    parser.add_argument('--dataset_dir', type=str, help='Directory where dataset is located')
    parser.add_argument('--model_dir', type=str, help='Directory to save trained model')
    parser.add_argument('--n_epochs', type=int, help='Number of training epochs', default=10)
    args = parser.parse_args()

    dataset = GymDataset(os.path.join(args.dataset_dir, f'{args.env}_dataset.pt'))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model_path = os.path.join(args.model_dir, f'{args.env}_sl_model.pth')
    model = RegressionNN(input_dim=dataset.in_dim(), output_dim=dataset.out_dim())
    model.learn(dataloader, epochs=args.n_epochs)
    model.save(model_path)

    # test if loading works
    model = RegressionNN(input_dim=dataset.in_dim(), output_dim=dataset.out_dim())
    model.load(model_path)