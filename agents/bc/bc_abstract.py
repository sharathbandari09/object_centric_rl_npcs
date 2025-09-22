"""
BC Abstract - Behavioral Cloning from entity-list
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


class BCAbstract(nn.Module):
    """
    Behavioral Cloning from entity-list observations.
    
    Architecture:
    - Per-entity encoder: MLP(F -> 64) with ReLU
    - Pooling: mean pooling across entities
    - Head: MLP(64 -> 64 -> n_actions) softmax
    """
    
    def __init__(self,
                 entity_dim: int = 9,
                 max_entities: int = 16,
                 n_actions: int = 6,
                 hidden_dim: int = 64):
        super(BCAbstract, self).__init__()
        
        self.entity_dim = entity_dim
        self.max_entities = max_entities
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        
        # Per-entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, entities: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            entities: Entity features (batch_size, max_entities, entity_dim)
            entity_mask: Entity mask (batch_size, max_entities)
            
        Returns:
            Action logits (batch_size, n_actions)
        """
        batch_size, max_entities, entity_dim = entities.shape
        
        # Reshape for per-entity processing
        entities_flat = entities.view(-1, entity_dim)  # (batch_size * max_entities, entity_dim)
        
        # Per-entity encoding
        entity_features = self.entity_encoder(entities_flat)  # (batch_size * max_entities, hidden_dim)
        
        # Reshape back
        entity_features = entity_features.view(batch_size, max_entities, self.hidden_dim)
        
        # Apply mask (set non-entities to zero)
        entity_mask_expanded = entity_mask.unsqueeze(-1).expand_as(entity_features)
        entity_features = entity_features * entity_mask_expanded.float()
        
        # Mean pooling across entities
        # Sum features and divide by number of valid entities
        pooled_features = torch.sum(entity_features, dim=1)  # (batch_size, hidden_dim)
        num_valid_entities = torch.sum(entity_mask, dim=1, keepdim=True).float()  # (batch_size, 1)
        
        # Avoid division by zero
        num_valid_entities = torch.clamp(num_valid_entities, min=1.0)
        pooled_features = pooled_features / num_valid_entities
        
        # Policy head
        action_logits = self.policy_head(pooled_features)
        
        return action_logits
    
    def get_action(self, entities: torch.Tensor, entity_mask: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Get action from observation.
        
        Args:
            entities: Entity features (batch_size, max_entities, entity_dim)
            entity_mask: Entity mask (batch_size, max_entities)
            deterministic: If True, return argmax action; if False, sample from policy
            
        Returns:
            Actions (batch_size,)
        """
        with torch.no_grad():
            action_logits = self.forward(entities, entity_mask)
            
            if deterministic:
                actions = torch.argmax(action_logits, dim=1)
            else:
                action_probs = torch.softmax(action_logits, dim=1)
                actions = torch.multinomial(action_probs, 1).squeeze(1)
            
            return actions


class BCAbstractTrainer:
    """
    Trainer for BC Abstract model.
    """
    
    def __init__(self,
                 model: BCAbstract,
                 learning_rate: float = 3e-4,
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
        
        for batch in tqdm(dataloader, desc="Training BC Abstract"):
            # Move to device
            entities = batch['entities'].to(self.device)
            entity_mask = batch['entity_mask'].to(self.device)
            actions = batch['action'].to(self.device)
            
            # Forward pass
            action_logits = self.model(entities, entity_mask)
            
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
                entities = batch['entities'].to(self.device)
                entity_mask = batch['entity_mask'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Forward pass
                action_logits = self.model(entities, entity_mask)
                
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
        
        print(f"Training BC Abstract for {epochs} epochs...")
        
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
                entities = batch['entities'].to(self.device)
                entity_mask = batch['entity_mask'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Get predictions
                action_logits = self.model(entities, entity_mask)
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
