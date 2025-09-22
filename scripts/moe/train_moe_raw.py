#!/usr/bin/env python3
"""
Train MoE with improved gating to fix expert utilization issues
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
import sys
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class MoERaw(nn.Module):
    """MoE Raw with improved gating and expert diversity"""
    
    def __init__(self, n_experts=4, input_dim=64, output_dim=6, device='cpu'):
        super().__init__()
        self.n_experts = n_experts
        self.device = device
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, input_dim),
            nn.ReLU()
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            ) for _ in range(n_experts)
        ])
        
        # Gating network with temperature scaling
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts)
        )
        
        # Temperature for gating (higher = more uniform)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Expert diversity loss weight (increased for better balance)
        self.diversity_weight = 0.5
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Gating with temperature
        gate_logits = self.gate(features) / self.temperature
        gate_weights = torch.softmax(gate_logits, dim=-1)
        
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(features))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, n_experts, output_dim)
        
        # Weighted combination
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)
        
        return output, gate_weights
    
    def diversity_loss(self, gate_weights):
        """Encourage expert diversity"""
        # Calculate entropy of gate weights
        entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1)
        max_entropy = np.log(self.n_experts)
        
        # Diversity loss (encourage high entropy)
        diversity_loss = -entropy.mean() / max_entropy
        
        return diversity_loss
    
    def load_balance_loss(self, gate_weights):
        """Load balancing loss to prevent expert collapse"""
        # Average gate weights across batch
        avg_gate_weights = gate_weights.mean(dim=0)
        
        # Encourage uniform distribution
        uniform_target = torch.ones_like(avg_gate_weights) / self.n_experts
        load_balance_loss = nn.MSELoss()(avg_gate_weights, uniform_target)
        
        return load_balance_loss

def convert_observations_to_pixels(observations):
    """Convert observations to pixel format for MoE Raw"""
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

def train_moe_raw():
    """Train MoE Raw model"""
    print("=== TRAINING MoE RAW ===")
    
    device = 'cpu'
    
    # Load dataset
    with open('datasets/raw_keydoor_dataset_new.pkl', 'rb') as f:
        data = pickle.load(f)
    
    observations = data['observations']
    actions = data['actions']
    
    print(f"Loaded dataset: {len(observations)} samples")
    
    # Convert observations to pixel format
    print("Converting observations to pixel format...")
    pixel_obs = convert_observations_to_pixels(observations)
    
    # Convert to tensors
    X = torch.tensor(pixel_obs, dtype=torch.float32)
    y = torch.tensor(actions, dtype=torch.long)
    
    # Create model
    model = MoERaw(n_experts=4, input_dim=64, output_dim=6, device=device)
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters (increased for better convergence)
    batch_size = 32
    n_epochs = 100
    
    # Training loop
    model.train()
    best_accuracy = 0.0
    
    for epoch in range(n_epochs):
        total_loss = 0
        total_accuracy = 0
        total_diversity_loss = 0
        total_load_balance_loss = 0
        n_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, len(X), batch_size):
            batch_X = X_shuffled[i:i+batch_size].to(device)
            batch_y = y_shuffled[i:i+batch_size].to(device)
            
            # Forward pass
            logits, gate_weights = model(batch_X)
            
            # Classification loss
            classification_loss = criterion(logits, batch_y)
            
            # Diversity loss
            diversity_loss = model.diversity_loss(gate_weights)
            
            # Load balance loss
            load_balance_loss = model.load_balance_loss(gate_weights)
            
            # Total loss
            total_loss_batch = (classification_loss + 
                              model.diversity_weight * diversity_loss + 
                              model.diversity_weight * load_balance_loss)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            # Metrics
            total_loss += total_loss_batch.item()
            total_diversity_loss += diversity_loss.item()
            total_load_balance_loss += load_balance_loss.item()
            
            # Accuracy
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch_y).float().mean()
            total_accuracy += accuracy.item()
            
            n_batches += 1
        
        # Log metrics
        avg_loss = total_loss / n_batches
        avg_accuracy = total_accuracy / n_batches
        avg_diversity_loss = total_diversity_loss / n_batches
        avg_load_balance_loss = total_load_balance_loss / n_batches
        
        # Save best model
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            save_dir = Path("experiments/exp_moe_raw_keydoor_seed0")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': avg_accuracy,
                'loss': avg_loss,
                'epoch': epoch
            }, save_dir / 'moe_raw_best.pth')
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}, "
                  f"Diversity={avg_diversity_loss:.4f}, LoadBalance={avg_load_balance_loss:.4f}")
            
            # Check gate weight distribution
            with torch.no_grad():
                sample_X = X[:100].to(device)
                _, sample_gate_weights = model(sample_X)
                avg_gate_weights = sample_gate_weights.mean(dim=0)
                print(f"Gate weights: {avg_gate_weights.cpu().numpy()}")
    
    # Save model
    save_dir = Path("experiments/exp_moe_raw_keydoor_seed0")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': avg_accuracy,
        'loss': avg_loss
    }, save_dir / 'moe_raw_final.pth')
    
    print(f"✅ MoE Raw model saved to {save_dir}")
    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    
    # Test gate weight distribution
    model.eval()
    with torch.no_grad():
        sample_X = X[:1000].to(device)
        _, gate_weights = model(sample_X)
        avg_gate_weights = gate_weights.mean(dim=0)
        gate_entropy = -torch.sum(avg_gate_weights * torch.log(avg_gate_weights + 1e-8))
        max_entropy = np.log(model.n_experts)
        
        print(f"\nFinal gate weight distribution: {avg_gate_weights.cpu().numpy()}")
        print(f"Gate entropy: {gate_entropy:.3f} / {max_entropy:.3f} ({gate_entropy/max_entropy:.3f})")
        
        # Check if experts are well utilized
        expert_utilization = (avg_gate_weights > 0.1).sum().item()
        print(f"Experts with >10% utilization: {expert_utilization}/{model.n_experts}")

if __name__ == '__main__':
    train_moe_raw()
