import torch
import torch.nn as nn
import torch.optim as optim
import os
import mlflow
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.utils import resample
import numpy as np
from .model_evaluation import evaluate_model_calibration

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {filepath}")

def nll_gaussian_loss(y_true, mean, log_var):
    # Variance is exp(log_variance) to ensure it's positive
    var = torch.exp(log_var)
    
    # Calculate NLL loss
    nll = 0.5 * torch.log(var) + (y_true - mean) ** 2 / (2 * var)
    
    # Return the mean NLL over all samples
    return torch.mean(nll)

def create_bootstrap_datasets(dataset, num_models):
    subset_indices = []
    for _ in range(num_models):
        indices = torch.randperm(len(dataset))[:len(dataset) // 2]  # Sample half the dataset
        subset_indices.append(indices)
    return [Subset(dataset, indices) for indices in subset_indices]

def train(model, train_dataset, epoch_number, batch_size=1024):
    # Split the dataset
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    
    mse_criterion = nn.MSELoss()
    device = next(model.parameters()).device

    model.train()  # Set model to training mode
    train_loss = 0.0
    train_mse = 0.0

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = model.calculate_loss(inputs, targets)
        prediction = model(inputs)
        if type(prediction) is tuple:
                prediction = prediction[0] # mean
        mse_loss = mse_criterion(prediction, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_mse += mse_loss.item()
    train_loss /= len(train_loader)
    train_mse /= len(train_loader)

    mlflow.log_metric(f"{model.label}_train_loss", train_loss, step=epoch_number)
    mlflow.log_metric(f"{model.label}_train_mse", train_mse, step=epoch_number)
    print(f"{model.label} - Epoch [{epoch_number + 1}] - "
              f"Train Loss: {train_loss:.4f} - Train MSE: {train_mse:.6f} - ")

def evaluate_model(model, test_dataset, epoch_number, batch_size=1024):
    # Evaluate on the test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    device = next(model.parameters()).device
    model.eval()
    test_mse = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
         for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            prediction = model(inputs)
            if type(prediction) is tuple:
                prediction = prediction[0] # mean
            mse_loss = criterion(prediction, targets)
            test_mse += mse_loss.item()
    test_mse /= len(test_loader)

    avg_ece, avg_coverage_1sigma, avg_coverage_2sigma = evaluate_model_calibration(model, test_loader, device=device)
    
    metrics = {"mse_test": test_mse, "ece": avg_ece, 
               "coverage_1sigma": avg_coverage_1sigma, "coverage_2sigma": avg_coverage_2sigma,
    }   
    # Log each metric to MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"{model.label}_{metric_name}", metric_value, step=epoch_number)

    # Print all metrics in a single line for easier reading
    print(f" Test Loss (MSE): {test_mse:.6f} - ECE: {avg_ece:.4f} - "
        f"Coverage within 1 sigma: {avg_coverage_1sigma * 100:.2f}% - Coverage within 2 sigma: {avg_coverage_2sigma * 100:.2f}%"
    )

class BayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, device='cpu', label='cost_model'):
        super(BayesianNN, self).__init__()
        # Define the layers
        self.label = label
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)  # Outputs the mean (μ)
        self.log_variance = nn.Linear(hidden_dim, output_dim)  # Outputs log(σ²) for stability
        if device:
            self.to(device)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)  # Predict mean (μ)
        log_var = self.log_variance(x)  # Predict log(σ²)
        
        return mean, log_var  # Return mean and log(variance)

    def calculate_loss(self, inputs, target):
        mean, log_var = self(inputs)
        loss = nll_gaussian_loss(target, mean, log_var) # Compute NLL loss
        return loss

    def learn(self, dataloader, epochs=10):
        train_dataset, test_dataset = torch.utils.data.random_split(dataloader.dataset, [0.8, 0.2])
        for epoch in range(epochs):
            train(self, train_dataset, epoch)
            evaluate_model(self, test_dataset, epoch, batch_size=1024)
    
    def save(self, directory_path, model_type):
        save_model(self, os.path.join(directory_path, f'{model_type}.pt'))
    
    def load(self, path):
        load_model(self, path)


class EnsembleModel(nn.Module):  # Subclassing torch.nn.Module
    def __init__(self, input_dim, output_dim, device='cpu', num_models=5, label='ensemble'):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList([RegressionNN(input_dim, output_dim) for _ in range(num_models)])
        self.device = device
        self.label=label
        if device:
            self.to(device)

    def forward(self, x):
        # Collect predictions from each model in the ensemble
        predictions = torch.stack([model(x) for model in self.models])
        mean_prediction = predictions.mean(dim=0)
        variance_prediction = predictions.var(dim=0)
        return mean_prediction, variance_prediction

    def learn(self, dataloader, epochs=10):
        
        train_dataset, test_dataset = torch.utils.data.random_split(dataloader.dataset, [0.8, 0.2])
        
        for epoch in range(epochs):
            subsets = create_bootstrap_datasets(train_dataset, len(self.models))
            for model, subset in zip(self.models, subsets):
                train(model, subset, epoch)
            evaluate_model(self, test_dataset, epoch, batch_size=1024)

    def save(self, directory_path, model_type):
        """
        Save an ensemble of models to a specified directory.
        """
        for i, model in enumerate(self.models):
            # Define a unique path for each model in the ensemble
            model_path = os.path.join(directory_path, f"{model_type}_{i+1}.pt")
            save_model(model, model_path)
        print(f"Ensemble saved to {directory_path}")

    def load(self, directory_path):
        """
        Load an ensemble of models from a specified directory.

        Args:
            directory_path (str): Path to the directory where models are saved.
        """
        for i, model in enumerate(self.models):
            # Define a unique path for each model in the ensemble
            model_path = os.path.join(directory_path, f"ensemble_model_{i+1}.pt")
            model.load_state_dict(torch.load(model_path, map_location=self.device))


class RegressionNN(nn.Module):
    def __init__(self, input_dim, output_dim, label='cost_model'):
        super(RegressionNN, self).__init__()
        self.label = label
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self._initialize_weights()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def calculate_mse_loss(self, inputs, target):
        criterion = nn.MSELoss()
        outputs = self(inputs)
        loss = criterion(outputs, target)  # Compute MSE loss
        return loss

    def calculate_loss(self, inputs, target):
        return self.calculate_mse_loss(inputs, target)
    
    def learn(self, dataloader, epochs=10):
        train(self, dataloader, epochs)

    def save(self, path):
        save_model(self, path)
    
    def load(self, path):
        load_model(self, path)
    
    def _initialize_weights(self):
        """
        Custom weight initialization for the network.
        Ensures random weights for each model instance.
        """
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)