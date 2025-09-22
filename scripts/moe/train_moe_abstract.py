#!/usr/bin/env python3
"""
Train MoE Abstract - Mixture of Experts with entity features
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

class MoEAbstract(nn.Module):
    """MoE Abstract - Mixture of Experts with entity features"""
    
    def __init__(self, n_experts=4, entity_dim=10, max_entities=16, n_actions=6, hidden_dim=128, device='cpu'):
        super().__init__()
        self.n_experts = n_experts
        self.entity_dim = entity_dim
        self.max_entities = max_entities
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Per-entity encoder (shared across experts)
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            ) for _ in range(n_experts)
        ])
        
        # Gating network with temperature scaling
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)
        )
        
        # Temperature for gating (higher = more uniform)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Expert diversity loss weight
        self.diversity_weight = 0.1
        
    def forward(self, entities, entity_mask=None):
        batch_size, max_entities, entity_dim = entities.shape
        
        # Encode entities
        entities_flat = entities.view(-1, entity_dim)
        entity_features = self.entity_encoder(entities_flat)
        entity_features = entity_features.view(batch_size, max_entities, self.hidden_dim)
        
        # Apply entity mask
        if entity_mask is not None:
            entity_mask_expanded = entity_mask.unsqueeze(-1).expand_as(entity_features)
            entity_features = entity_features * entity_mask_expanded.float()
        
        # Mean pooling over entities
        pooled_features = torch.sum(entity_features, dim=1)
        
        # Gating with temperature
        gate_logits = self.gate(pooled_features) / self.temperature
        gate_weights = torch.softmax(gate_logits, dim=-1)
        
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(pooled_features))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, n_experts, n_actions)
        
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

def train_moe_abstract():
    """Train MoE Abstract model"""
    print("=== TRAINING MoE ABSTRACT ===")
    
    device = 'cpu'
    
    # Load entity dataset
    with open('datasets/entity_keydoor_dataset_new.pkl', 'rb') as f:
        data = pickle.load(f)
    
    observations = data['observations']
    actions = data['actions']
    
    # Convert to tensors
    entities_list = []
    actions_list = []
    masks_list = []
    
    for obs, action in zip(observations, actions):
        entities_list.append(obs['entities'])
        actions_list.append(action)
        masks_list.append(obs['entity_mask'])
    
    entities = torch.tensor(np.array(entities_list), dtype=torch.float32)
    actions = torch.tensor(actions_list, dtype=torch.long)
    masks = torch.tensor(np.array(masks_list), dtype=torch.bool)
    
    print(f"Loaded dataset: {len(observations)} samples")
    print(f"Entities shape: {entities.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Masks shape: {masks.shape}")
    
    # Create model
    model = MoEAbstract(n_experts=4, entity_dim=10, max_entities=16, n_actions=6, hidden_dim=128, device=device)
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    batch_size = 64
    n_epochs = 50
    
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
        indices = torch.randperm(len(entities))
        entities_shuffled = entities[indices]
        actions_shuffled = actions[indices]
        masks_shuffled = masks[indices]
        
        for i in range(0, len(entities), batch_size):
            batch_entities = entities_shuffled[i:i+batch_size].to(device)
            batch_actions = actions_shuffled[i:i+batch_size].to(device)
            batch_masks = masks_shuffled[i:i+batch_size].to(device)
            
            # Forward pass
            logits, gate_weights = model(batch_entities, batch_masks)
            
            # Classification loss
            classification_loss = criterion(logits, batch_actions)
            
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
            accuracy = (predictions == batch_actions).float().mean()
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
            save_dir = Path("experiments/exp_moe_abstract_keydoor_seed0")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': avg_accuracy,
                'loss': avg_loss,
                'epoch': epoch
            }, save_dir / 'moe_abstract_best.pth')
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}, "
                  f"Diversity={avg_diversity_loss:.4f}, LoadBalance={avg_load_balance_loss:.4f}")
            
            # Check gate weight distribution
            with torch.no_grad():
                sample_entities = entities[:100].to(device)
                sample_masks = masks[:100].to(device)
                _, sample_gate_weights = model(sample_entities, sample_masks)
                avg_gate_weights = sample_gate_weights.mean(dim=0)
                print(f"Gate weights: {avg_gate_weights.cpu().numpy()}")
    
    # Save model
    save_dir = Path("experiments/exp_moe_abstract_keydoor_seed0")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': avg_accuracy,
        'loss': avg_loss
    }, save_dir / 'moe_abstract_final.pth')
    
    print(f"✅ MoE Abstract model saved to {save_dir}")
    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    
    # Test gate weight distribution
    model.eval()
    with torch.no_grad():
        sample_entities = entities[:1000].to(device)
        sample_masks = masks[:1000].to(device)
        _, gate_weights = model(sample_entities, sample_masks)
        avg_gate_weights = gate_weights.mean(dim=0)
        gate_entropy = -torch.sum(avg_gate_weights * torch.log(avg_gate_weights + 1e-8))
        max_entropy = np.log(model.n_experts)
        
        print(f"\nFinal gate weight distribution: {avg_gate_weights.cpu().numpy()}")
        print(f"Gate entropy: {gate_entropy:.3f} / {max_entropy:.3f} ({gate_entropy/max_entropy:.3f})")
        
        # Check if experts are well utilized
        expert_utilization = (avg_gate_weights > 0.1).sum().item()
        print(f"Experts with >10% utilization: {expert_utilization}/{model.n_experts}")

if __name__ == '__main__':
    train_moe_abstract()


