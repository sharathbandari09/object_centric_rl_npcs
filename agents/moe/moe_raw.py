"""
MoE Raw - Mixture of Experts from pixel observations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os
import sys
from tqdm import tqdm

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# from datasets.demo_dataset import DemoDataset  # Not needed for the model class


class MoERaw(nn.Module):
    """
    Mixture of Experts from pixel observations.
    
    Architecture:
    - Shared conv encoder: Conv(1,16,3) -> ReLU -> Conv(16,32,3) -> ReLU -> Conv(32,64,3) -> ReLU
    - Flatten -> FC(64*H*W, 256) -> ReLU -> FC(256, 64) -> ReLU
    - Gating network: FC(64, K) -> softmax (top-k routing)
    - K expert networks: each FC(64, n_actions)
    - Output: weighted combination of expert outputs
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (1, 8, 8),
                 n_actions: int = 6,
                 n_experts: int = 4,
                 hidden_dim: int = 256,
                 latent_dim: int = 64,
                 expert_dim: int = 32,
                 top_k: int = 2):
        super(MoERaw, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.expert_dim = expert_dim
        self.top_k = top_k
        
        # Shared conv encoder
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size after conv layers
        self.conv_output_size = self._get_conv_output_size()
        
        # Shared MLP layers
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Gating network
        self.gating_network = nn.Linear(latent_dim, n_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, n_actions)
            ) for _ in range(n_experts)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _get_conv_output_size(self) -> int:
        """Calculate the output size after conv layers."""
        dummy_input = torch.zeros(1, *self.input_shape)
        with torch.no_grad():
            conv_output = self.conv_encoder(dummy_input)
            return int(np.prod(conv_output.shape[1:]))
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pixel_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pixel_obs: Pixel observations (batch_size, 1, H, W)
            
        Returns:
            Tuple of (action_logits, gate_probs)
        """
        # Conv encoding
        conv_features = self.conv_encoder(pixel_obs)
        
        # Flatten
        flattened = conv_features.view(conv_features.size(0), -1)
        
        # Shared MLP
        shared_features = self.shared_mlp(flattened)
        
        # Gating network
        gate_logits = self.gating_network(shared_features)
        gate_probs = torch.softmax(gate_logits, dim=1)
        
        # Top-k routing
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)  # Renormalize
        
        # Expert outputs
        expert_outputs = []
        for i in range(self.n_experts):
            expert_output = self.experts[i](shared_features)
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, n_experts, n_actions)
        
        # Weighted combination
        batch_size = pixel_obs.size(0)
        action_logits = torch.zeros(batch_size, self.n_actions, device=pixel_obs.device)
        
        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j]
                weight = top_k_probs[i, j]
                action_logits[i] += weight * expert_outputs[i, expert_idx]
        
        return action_logits, gate_probs
    
    def get_action(self, pixel_obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Get action from observation.
        
        Args:
            pixel_obs: Pixel observations (batch_size, 1, H, W)
            deterministic: If True, return argmax action; if False, sample from policy
            
        Returns:
            Actions (batch_size,)
        """
        with torch.no_grad():
            action_logits, _ = self.forward(pixel_obs)
            
            if deterministic:
                actions = torch.argmax(action_logits, dim=1)
            else:
                action_probs = torch.softmax(action_logits, dim=1)
                actions = torch.multinomial(action_probs, 1).squeeze(1)
            
            return actions
    
    def get_gate_usage(self, pixel_obs: torch.Tensor) -> torch.Tensor:
        """Get gate usage statistics."""
        with torch.no_grad():
            _, gate_probs = self.forward(pixel_obs)
            return gate_probs


class MoERawTrainer:
    """
    Trainer for MoE Raw model.
    """
    
    def __init__(self,
                 model: MoERaw,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 load_balance_coef: float = 0.1,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.load_balance_coef = load_balance_coef
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.load_balance_losses = []
    
    def compute_load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage balanced expert usage.
        
        Args:
            gate_probs: Gate probabilities (batch_size, n_experts)
            
        Returns:
            Load balance loss
        """
        # Compute mean gate usage per expert
        mean_gate_usage = torch.mean(gate_probs, dim=0)  # (n_experts,)
        
        # Target is uniform distribution
        target_usage = torch.ones_like(mean_gate_usage) / self.model.n_experts
        
        # Load balance loss (KL divergence to uniform)
        load_balance_loss = torch.sum(mean_gate_usage * torch.log(mean_gate_usage / target_usage))
        
        return load_balance_loss
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_load_balance_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training MoE Raw"):
            # Move to device
            pixel_obs = batch['pixel'].to(self.device)
            actions = batch['action'].to(self.device)
            
            # Forward pass
            action_logits, gate_probs = self.model(pixel_obs)
            
            # Compute losses
            action_loss = self.criterion(action_logits, actions)
            load_balance_loss = self.compute_load_balance_loss(gate_probs)
            
            # Total loss
            total_loss_batch = action_loss + self.load_balance_coef * load_balance_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += action_loss.item()
            total_load_balance_loss += load_balance_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_load_balance_loss = total_load_balance_loss / num_batches
        
        self.train_losses.append(avg_loss)
        self.load_balance_losses.append(avg_load_balance_loss)
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                pixel_obs = batch['pixel'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Forward pass
                action_logits, _ = self.model(pixel_obs)
                
                # Compute loss
                loss = self.criterion(action_logits, actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              epochs: int = 100,
              patience: int = 10,
              save_path: str = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training MoE Raw for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Validate
            val_loss = self.validate(val_dataloader)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                  f"Load Balance Loss = {self.load_balance_losses[-1]:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if save_path:
                    self.save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'load_balance_losses': self.load_balance_losses,
            'best_val_loss': best_val_loss
        }
    
    def save_model(self, path: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'load_balance_losses': self.load_balance_losses
        }, path)
    
    def load_model(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.load_balance_losses = checkpoint['load_balance_losses']
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        gate_usage_stats = []
        
        with torch.no_grad():
            for batch in dataloader:
                pixel_obs = batch['pixel'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Get predictions
                action_logits, gate_probs = self.model(pixel_obs)
                predicted_actions = torch.argmax(action_logits, dim=1)
                
                # Calculate accuracy
                total_correct += (predicted_actions == actions).sum().item()
                total_samples += actions.size(0)
                
                # Collect gate usage stats
                gate_usage_stats.append(gate_probs.mean(dim=0))
        
        accuracy = total_correct / total_samples
        
        # Compute gate usage statistics
        gate_usage_stats = torch.stack(gate_usage_stats).mean(dim=0)
        
        return {
            'accuracy': accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples,
            'gate_usage': gate_usage_stats.cpu().numpy()
        }











