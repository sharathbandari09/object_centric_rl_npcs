"""
BC Raw - Behavioral Cloning from pixels
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

# Now import the dataset module
from datasets.demo_dataset import DemoDataset


class BCRaw(nn.Module):
    """
    Behavioral Cloning from pixel observations.
    
    Architecture:
    - Conv encoder: Conv(1,16,3) -> ReLU -> Conv(16,32,3) -> ReLU -> Conv(32,64,3) -> ReLU
    - Flatten -> FC(64*H*W, 256) -> ReLU -> FC(256, 64) -> ReLU
    - Head: FC(64, n_actions) -> softmax
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (1, 8, 8),
                 n_actions: int = 6,
                 hidden_dim: int = 256,
                 latent_dim: int = 64):
        super(BCRaw, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Conv encoder
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
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Linear(latent_dim, n_actions)
        
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
    
    def forward(self, pixel_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_obs: Pixel observations (batch_size, 1, H, W)
            
        Returns:
            Action logits (batch_size, n_actions)
        """
        # Conv encoding
        conv_features = self.conv_encoder(pixel_obs)
        
        # Flatten
        flattened = conv_features.view(conv_features.size(0), -1)
        
        # MLP
        latent = self.mlp(flattened)
        
        # Policy head
        action_logits = self.policy_head(latent)
        
        return action_logits
    
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
            action_logits = self.forward(pixel_obs)
            
            if deterministic:
                actions = torch.argmax(action_logits, dim=1)
            else:
                action_probs = torch.softmax(action_logits, dim=1)
                actions = torch.multinomial(action_probs, 1).squeeze(1)
            
            return actions


class BCRawTrainer:
    """
    Trainer for BC Raw model.
    """
    
    def __init__(self,
                 model: BCRaw,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
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
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training BC Raw"):
            # Move to device
            pixel_obs = batch['pixel'].to(self.device)
            actions = batch['action'].to(self.device)
            
            # Forward pass
            action_logits = self.model(pixel_obs)
            
            # Compute loss
            loss = self.criterion(action_logits, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
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
                action_logits = self.model(pixel_obs)
                
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
        
        print(f"Training BC Raw for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Validate
            val_loss = self.validate(val_dataloader)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
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
            'best_val_loss': best_val_loss
        }
    
    def save_model(self, path: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_model(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                pixel_obs = batch['pixel'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Get predictions
                action_logits = self.model(pixel_obs)
                predicted_actions = torch.argmax(action_logits, dim=1)
                
                # Calculate accuracy
                total_correct += (predicted_actions == actions).sum().item()
                total_samples += actions.size(0)
        
        accuracy = total_correct / total_samples
        
        return {
            'accuracy': accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples
        }
