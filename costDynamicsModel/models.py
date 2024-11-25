import torch
import torch.nn as nn
import torch.optim as optim
import os

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


class BayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(BayesianNN, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)  # Outputs the mean (μ)
        self.log_variance = nn.Linear(hidden_dim, output_dim)  # Outputs log(σ²) for stability
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)  # Predict mean (μ)
        log_var = self.log_variance(x)  # Predict log(σ²)
        
        return mean, log_var  # Return mean and log(variance)

    def learn(self, dataloader, epochs=10):        
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):  # Example with 100 epochs
            self.train()  # Set model to training mode
            running_loss = 0.0
            for batch_idx, (inputs, target) in enumerate(dataloader):
                optimizer.zero_grad()
                mean, log_var = self(inputs)
                loss = nll_gaussian_loss(target, mean, log_var) # Compute NLL loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')

    def save(self, path):
        save_model(self, path)
    
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
    def __init__(self, input_dim, output_dim):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def learn(self, dataloader, epochs=10):
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            for batch_idx, (observations, target_values) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self(observations)
                loss = criterion(outputs, target_values)  # Compute MSE loss
                loss.backward()
                optimizer.step()
                
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def save(self, path):
        save_model(self, path)
    
    def load(self, path):
        load_model(self, path)