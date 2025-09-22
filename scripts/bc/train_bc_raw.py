#!/usr/bin/env python3
"""
Train BC Raw model (pixel-based) for comparison with object-centric methods
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
import argparse
import os

# Project root
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

class BCRaw(nn.Module):
    def __init__(self, input_channels=1, max_grid_size=12, n_actions=6, hidden_dim=128):
        super().__init__()
        # Simple CNN for pixel input - handles variable grid sizes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Always reduce to 4x4 regardless of input size
        )
        
        # Calculate flattened size
        self.flattened_size = 64 * 4 * 4
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, pixel_obs):
        # pixel_obs shape: (batch_size, 1, grid_size, grid_size)
        conv_out = self.conv_layers(pixel_obs)
        flattened = conv_out.view(conv_out.size(0), -1)
        action_logits = self.policy_head(flattened)
        return action_logits

def load_raw_dataset():
    """Load the raw dataset with pixel observations"""
    print("Loading raw dataset...")
    
    # Try new dataset first, fallback to old format
    dataset_files = [
        'datasets/raw_keydoor_dataset_new.pkl',
        'datasets/raw_keydoor_dataset.pkl'
    ]
    
    for dataset_file in dataset_files:
        if os.path.exists(dataset_file):
            with open(dataset_file, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different dataset formats
            if 'pixel_observations' in data:
                # Old format
                pixel_observations = data['pixel_observations']
                actions = data['actions']
            else:
                # New format - need to convert observations to pixel format
                observations = data['observations']
                actions = data['actions']
                pixel_observations = convert_observations_to_pixels(observations)
            
            print(f"Loaded raw dataset from {dataset_file}:")
            print(f"  Pixel observations: {pixel_observations.shape}")
            print(f"  Actions: {len(actions)} (list format)")
            
            return pixel_observations, actions
    
    raise FileNotFoundError("No raw dataset found. Please run dataset generation first.")

def convert_observations_to_pixels(observations):
    """Convert new observation format to pixel format - handles variable grid sizes"""
    pixel_obs = []
    
    for obs in observations:
        # Extract grid - can be different sizes (6x6, 8x8, 9x9, 10x10)
        grid = obs['grid']  # Shape: (grid_size, grid_size)
        grid_size = grid.shape[0]
        
        # Pad smaller grids to 12x12 for consistent model input
        max_size = 12
        pixel_grid = np.zeros((1, max_size, max_size), dtype=np.float32)
        
        # Copy the actual grid data to the top-left corner
        for r in range(grid_size):
            for c in range(grid_size):
                cell_type = grid[r, c]
                if cell_type == 1:  # Wall
                    pixel_grid[0, r, c] = 1.0
                elif cell_type == 2:  # Agent
                    pixel_grid[0, r, c] = 0.5
                elif cell_type == 3:  # Key
                    pixel_grid[0, r, c] = 0.8
                elif cell_type == 4:  # Door
                    pixel_grid[0, r, c] = 0.3
                # Floor (0) remains 0.0
        
        pixel_obs.append(pixel_grid)
    
    return np.array(pixel_obs)

def main():
    parser = argparse.ArgumentParser(description="Train BC Raw model.")
    parser.add_argument('--output_dir', type=str, default='experiments/exp_bc_raw_keydoor_seed0',
                        help='Output directory for the model')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cpu'
    
    # Load raw dataset
    pixel_obs, actions = load_raw_dataset()
    print(f"Loaded raw dataset: {pixel_obs.shape} observations, {len(actions)} actions")
    
    # Convert to tensors
    pixel_tensor = torch.tensor(pixel_obs, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(pixel_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model, optimizer, loss
    model = BCRaw().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_pixels, batch_actions in dataloader:
            optimizer.zero_grad()
            logits = model(batch_pixels)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += batch_actions.size(0)
            correct_predictions += (predicted == batch_actions).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': accuracy,
                'loss': avg_loss
            }, os.path.join(args.output_dir, 'bc_raw_best.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'accuracy': accuracy,
        'loss': avg_loss
    }, os.path.join(args.output_dir, 'bc_raw_final.pth'))
    
    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    print(f"Models saved to {args.output_dir}")

if __name__ == '__main__':
    main()
