import torch
import numpy as np


def expected_calibration_error(y_true, mean, var, num_bins=10):
    std_dev = torch.sqrt(var)
    abs_error = torch.abs(y_true - mean)
    sorted_indices = torch.argsort(std_dev)
    bin_size = len(y_true) // num_bins
    
    ece = 0.0
    for i in range(num_bins):
        bin_indices = sorted_indices[i * bin_size: (i + 1) * bin_size]
        bin_pred_std = std_dev[bin_indices].mean()
        bin_empirical_error = abs_error[bin_indices].mean()
        ece += torch.abs(bin_pred_std - bin_empirical_error)
    
    ece /= num_bins
    return ece.item()

def coverage_probability(y_true, mean, var, sigma=1):
    std_dev = torch.sqrt(var)
    within_interval = (y_true >= mean - sigma * std_dev) & (y_true <= mean + sigma * std_dev)
    return within_interval.float().mean().item()

# Use these functions to calculate calibration metrics during evaluation
def evaluate_model_calibration(model, test_loader, device='cpu'):
    model.eval()
    with torch.no_grad():
        total_ece = 0.0
        coverage_within_1sigma = 0.0
        coverage_within_2sigma = 0.0
        num_batches = 0
        
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mean, log_var = model(inputs)
            
            # Observations
            obs_var = torch.exp(log_var)
            total_ece += expected_calibration_error(targets, mean, obs_var)
            coverage_within_1sigma += coverage_probability(targets, mean, obs_var, sigma=1)
            coverage_within_2sigma += coverage_probability(targets, mean, obs_var, sigma=2)
            
            # Repeat similarly for cost if needed

            num_batches += 1
        
        avg_ece = total_ece / num_batches
        avg_coverage_1sigma = coverage_within_1sigma / num_batches
        avg_coverage_2sigma = coverage_within_2sigma / num_batches
    
    return avg_ece, avg_coverage_1sigma, avg_coverage_2sigma

        
