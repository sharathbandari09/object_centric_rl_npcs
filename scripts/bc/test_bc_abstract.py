#!/usr/bin/env python3
"""
Test BC Abstract model with fixed entity features
"""
import torch
import numpy as np
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from envs.keydoor.keydoor_env import KeyDoorEnv

class AttentionPool(torch.nn.Module):
    """Attention pooling for entity features"""
    def __init__(self, entity_dim, hidden_dim=64):
        super().__init__()
        self.query = torch.nn.Linear(entity_dim, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, 1)
        
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

class BCAbstract(torch.nn.Module):
    def __init__(self, entity_dim=10, max_entities=16, n_actions=6, hidden_dim=256):
        super().__init__()
        self.entity_encoder = torch.nn.Sequential(
            torch.nn.Linear(entity_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
        # Attention pooling
        self.attn_pool = AttentionPool(hidden_dim, hidden_dim)
        
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_actions)
        )
        
        self.max_entities = max_entities
        self.hidden_dim = hidden_dim

    def forward(self, entities, mask):
        batch_size, max_entities, entity_dim = entities.shape
        entities_flat = entities.view(-1, entity_dim)
        entity_features = self.entity_encoder(entities_flat)
        entity_features = entity_features.view(batch_size, max_entities, self.hidden_dim)
        pooled = self.attn_pool(entity_features, mask)
        return self.policy_head(pooled)

def test_bc_abstract():
    print("=== Testing BC Abstract Model ===")
    
    device = 'cpu'
    
    # Load model
    model = BCAbstract(entity_dim=10, max_entities=16, n_actions=6).to(device)
    checkpoint = torch.load(os.path.join(project_root, 'experiments/exp_bc_abstract_keydoor_seed0/bc_abstract_best.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded with accuracy: {checkpoint['accuracy']:.4f}")
    
    env = KeyDoorEnv(max_steps=100)
    
    # Test on training layouts (T1-T6)
    print("\n--- Training Layouts (T1-T6) ---")
    training_success = 0
    training_total = 0
    
    for template_id in range(1, 7):  # T1-T6
        print(f"Testing template {template_id}...")
        template_success = 0
        
        for i in range(10):  # 10 episodes per template
            obs = env.reset(template_id=template_id, seed=i)
            
            done = False
            episode_reward = 0
            
            for t in range(100):  # Max steps
                # Get model prediction
                entities_t = torch.from_numpy(obs['entities']).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(obs['entity_mask']).bool().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(entities_t, mask_t)
                    action = torch.argmax(logits, dim=-1).item()
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            if info.get('success', False):
                template_success += 1
                training_success += 1
            training_total += 1
        
        print(f"  Template {template_id}: {template_success}/10 successes")
    
    training_rate = training_success / training_total
    print(f"Training Layouts Success Rate: {training_rate:.2f} ({training_success}/{training_total})")
    
    # Test on novel layouts (T7-T10)
    print("\n--- Novel Layouts (T7-T10) ---")
    novel_success = 0
    novel_total = 0
    
    for template_id in range(7, 11):  # T7-T10
        print(f"Testing template {template_id}...")
        template_success = 0
        
        for i in range(10):  # 10 episodes per template
            obs = env.reset(template_id=template_id, seed=i)
            
            done = False
            episode_reward = 0
            
            for t in range(100):  # Max steps
                # Get model prediction
                entities_t = torch.from_numpy(obs['entities']).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(obs['entity_mask']).bool().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(entities_t, mask_t)
                    action = torch.argmax(logits, dim=-1).item()
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            if info.get('success', False):
                template_success += 1
                novel_success += 1
            novel_total += 1
        
        print(f"  Template {template_id}: {template_success}/10 successes")
    
    novel_rate = novel_success / novel_total
    print(f"Novel Layouts Success Rate: {novel_rate:.2f} ({novel_success}/{novel_total})")
    
    print(f"\n=== BC ABSTRACT RESULTS ===")
    print(f"Training Layouts (T1-T5): {training_rate:.2f} ({training_success}/{training_total})")
    print(f"Novel Layouts (T6-T7): {novel_rate:.2f} ({novel_success}/{novel_total})")
    print(f"Expected: Better generalization than raw methods")
    print(f"Actual: {training_rate:.0%} on training, {novel_rate:.0%} on novel")
    
    env.close()
    return training_rate, novel_rate

if __name__ == '__main__':
    test_bc_abstract()


