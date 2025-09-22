"""
MoE Abstract - Mixture of Experts from entity-list observations
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


class MoEAbstract(nn.Module):
    """
    Mixture of Experts from entity-list observations.
    
    Architecture:
    - Per-entity encoder: MLP(F -> 64) with ReLU
    - Pooling: mean pooling across entities
    - Gating network: FC(64, K) -> softmax (top-k routing)
    - K expert networks: each FC(64, n_actions)
    - Output: weighted combination of expert outputs
    """
    
    def __init__(self,
                 entity_dim: int = 9,
                 max_entities: int = 16,
                 n_actions: int = 6,
                 n_experts: int = 4,
                 hidden_dim: int = 64,
                 expert_dim: int = 32,
                 top_k: int = 2):
        super(MoEAbstract, self).__init__()
        
        self.entity_dim = entity_dim
        self.max_entities = max_entities
        self.n_actions = n_actions
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.top_k = top_k
        
        # Per-entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Gating network
        self.gating_network = nn.Linear(hidden_dim, n_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, n_actions)
            ) for _ in range(n_experts)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, entities: torch.Tensor, entity_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            entities: Entity features (batch_size, max_entities, entity_dim)
            entity_mask: Entity mask (batch_size, max_entities)
            
        Returns:
            Tuple of (action_logits, gate_probs)
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
        
        # Gating network
        gate_logits = self.gating_network(pooled_features)
        gate_probs = torch.softmax(gate_logits, dim=1)
        
        # Top-k routing
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)  # Renormalize
        
        # Expert outputs
        expert_outputs = []
        for i in range(self.n_experts):
            expert_output = self.experts[i](pooled_features)
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, n_experts, n_actions)
        
        # Weighted combination
        action_logits = torch.zeros(batch_size, self.n_actions, device=entities.device)
        
        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j]
                weight = top_k_probs[i, j]
                action_logits[i] += weight * expert_outputs[i, expert_idx]
        
        return action_logits, gate_probs
    
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
            action_logits, _ = self.forward(entities, entity_mask)
            
            if deterministic:
                actions = torch.argmax(action_logits, dim=1)
            else:
                action_probs = torch.softmax(action_logits, dim=1)
                actions = torch.multinomial(action_probs, 1).squeeze(1)
            
            return actions
    
    def get_gate_usage(self, entities: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        """Get gate usage statistics."""
        with torch.no_grad():
            _, gate_probs = self.forward(entities, entity_mask)
            return gate_probs


class MoEAbstractTrainer:
    """
    Trainer for MoE Abstract model.
    """
    
    def __init__(self,
                 model: MoEAbstract,
                 learning_rate: float = 3e-4,
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
        
        for batch in tqdm(dataloader, desc="Training MoE Abstract"):
            # Move to device
            entities = batch['entities'].to(self.device)
            entity_mask = batch['entity_mask'].to(self.device)
            actions = batch['action'].to(self.device)
            
            # Forward pass
            action_logits, gate_probs = self.model(entities, entity_mask)
            
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
                entities = batch['entities'].to(self.device)
                entity_mask = batch['entity_mask'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Forward pass
                action_logits, _ = self.model(entities, entity_mask)
                
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
        
        print(f"Training MoE Abstract for {epochs} epochs...")
        
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
                entities = batch['entities'].to(self.device)
                entity_mask = batch['entity_mask'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Get predictions
                action_logits, gate_probs = self.model(entities, entity_mask)
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











