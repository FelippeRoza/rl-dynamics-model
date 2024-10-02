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