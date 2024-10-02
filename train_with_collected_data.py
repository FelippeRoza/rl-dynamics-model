import torch
import torch.nn as nn
import torch.optim as optim
from costDynamicsModel import CDM
import continuousSafetyGym
from torch.utils.data import DataLoader, Dataset
import os
import argparse

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
    

class SafetyLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SafetyLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(dataloader, model, epochs=10):
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for batch_idx, (observations, target_values) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(observations)
            loss = criterion(outputs, target_values)  # Compute MSE loss
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {filepath}")


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
    model = SafetyLayerNN(input_dim=dataset.in_dim(), output_dim=dataset.out_dim())
    train_model(dataloader, model, epochs=args.n_epochs)
    save_model(model, model_path)

    # test if loading works
    model = SafetyLayerNN(input_dim=dataset.in_dim(), output_dim=dataset.out_dim())
    load_model(model, model_path)