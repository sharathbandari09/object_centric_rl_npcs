#!/usr/bin/env python3
"""
Train BC Abstract model (entity-based) for comparison with object-centric methods
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

class AttentionPool(nn.Module):
    """Attention pooling for entity features"""
    def __init__(self, entity_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Linear(entity_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, entities, mask):
        # entities: (batch_size, max_entities, entity_dim)
        # mask: (batch_size, max_entities)
        
        # Compute attention weights
        attn_weights = self.proj(torch.tanh(self.query(entities)))  # (batch_size, max_entities, 1)
        attn_weights = attn_weights.squeeze(-1)  # (batch_size, max_entities)
        
        # Apply mask
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Weighted sum
        pooled = torch.sum(entities * attn_weights.unsqueeze(-1), dim=1)  # (batch_size, entity_dim)
        return pooled

class BCAbstract(nn.Module):
    def __init__(self, entity_dim=10, max_entities=16, n_actions=6, hidden_dim=256):
        super().__init__()
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention pooling
        self.attn_pool = AttentionPool(hidden_dim, hidden_dim)
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        self.max_entities = max_entities
        self.hidden_dim = hidden_dim

    def forward(self, entities, mask):
        batch_size, max_entities, entity_dim = entities.shape
        entities_flat = entities.view(-1, entity_dim)
        entity_features = self.entity_encoder(entities_flat)
        entity_features = entity_features.view(batch_size, max_entities, self.hidden_dim)
        
        # Apply mask
        entity_features = entity_features * mask.unsqueeze(-1).float()
        
        # Attention pooling
        pooled_features = self.attn_pool(entity_features, mask)
        
        action_logits = self.policy_head(pooled_features)
        return action_logits

def load_entity_dataset():
    """Load entity dataset"""
    try:
        with open('datasets/entity_keydoor_dataset_new.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Convert to tensors
        entities_list = []
        actions_list = []
        masks_list = []
        
        for obs, action in zip(data['observations'], data['actions']):
            entities_list.append(obs['entities'])
            actions_list.append(action)
            masks_list.append(obs['entity_mask'])
        
        entities = torch.tensor(np.array(entities_list), dtype=torch.float32)
        actions = torch.tensor(actions_list, dtype=torch.long)
        masks = torch.tensor(np.array(masks_list), dtype=torch.bool)
        
        return entities, actions, masks
    
    except FileNotFoundError:
        print("Entity dataset not found. Please run dataset generation first.")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Train BC Abstract model.")
    parser.add_argument('--output_dir', type=str, default='experiments/exp_bc_abstract_keydoor_seed0',
                        help='Output directory for the model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cpu'
    
    # Load entity dataset
    entities, actions, masks = load_entity_dataset()
    if entities is None:
        return
    
    print(f"Loaded entity dataset: {entities.shape} entities, {len(actions)} actions")
    
    # Create dataset and dataloader
    dataset = TensorDataset(entities, actions, masks)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model, optimizer, loss
    model = BCAbstract(entity_dim=10, max_entities=16, n_actions=6, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_entities, batch_actions, batch_masks in dataloader:
            optimizer.zero_grad()
            logits = model(batch_entities, batch_masks)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += batch_actions.size(0)
            correct_predictions += (predicted == batch_actions).sum().item()
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy
            }, os.path.join(args.output_dir, 'bc_abstract_best.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs - 1,
        'loss': avg_loss,
        'accuracy': accuracy
    }, os.path.join(args.output_dir, 'bc_abstract_final.pth'))
    
    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    print(f"Models saved to {args.output_dir}")

if __name__ == "__main__":
    main()
