#!/usr/bin/env python3
"""
Test the MoE Abstract model
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
from agents.moe.moe_abstract import MoEAbstract

def test_moe_abstract():
    """Test MoE Abstract model"""
    print("=== TESTING MoE ABSTRACT ===")
    
    device = 'cpu'
    
    # Load model
    model = MoEAbstract(n_experts=4, entity_dim=10, max_entities=16, n_actions=6, hidden_dim=128, device=device)
    checkpoint = torch.load(os.path.join(project_root, 'experiments/exp_moe_abstract_keydoor_seed0/moe_abstract_final.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model accuracy: {checkpoint['accuracy']:.4f}")
    
    env = KeyDoorEnv(max_steps=100)
    
    # Test on training templates
    print("\n--- Testing on Training Templates ---")
    training_success = 0
    training_total = 0
    
    for template_id in range(1, 7):  # T1-T6
        template_success = 0
        template_total = 10
        
        for episode in range(template_total):
            obs = env.reset(template_id=template_id, seed=episode)
            
            done = False
            step = 0
            success = False
            
            while not done and step < 100:
                # Get prediction
                entities_t = torch.from_numpy(obs['entities']).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(obs['entity_mask']).bool().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, gate_weights = model(entities_t, mask_t)
                    action = torch.argmax(logits, dim=-1).item()
                
                obs, reward, done, info = env.step(action)
                step += 1
                
                if done and reward > 0:
                    success = True
                    break
            
            if success:
                template_success += 1
                training_success += 1
            
            training_total += 1
        
        success_rate = template_success / template_total
        print(f"Template {template_id}: {template_success}/{template_total} ({success_rate:.1%})")
    
    training_success_rate = training_success / training_total
    print(f"Training Success Rate: {training_success}/{training_total} ({training_success_rate:.1%})")
    
    # Test on novel templates
    print("\n--- Testing on Novel Templates ---")
    novel_success = 0
    novel_total = 0
    
    for template_id in range(7, 11):  # T7-T10
        template_success = 0
        template_total = 10
        
        for episode in range(template_total):
            obs = env.reset(template_id=template_id, seed=episode)
            
            done = False
            step = 0
            success = False
            
            while not done and step < 100:
                # Get prediction
                entities_t = torch.from_numpy(obs['entities']).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(obs['entity_mask']).bool().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, gate_weights = model(entities_t, mask_t)
                    action = torch.argmax(logits, dim=-1).item()
                
                obs, reward, done, info = env.step(action)
                step += 1
                
                if done and reward > 0:
                    success = True
                    break
            
            if success:
                template_success += 1
                novel_success += 1
            
            novel_total += 1
        
        success_rate = template_success / template_total
        print(f"Template {template_id}: {template_success}/{template_total} ({success_rate:.1%})")
    
    novel_success_rate = novel_success / novel_total
    print(f"Novel Success Rate: {novel_success}/{novel_total} ({novel_success_rate:.1%})")
    
    env.close()
    
    return {
        'training_success_rate': training_success_rate,
        'novel_success_rate': novel_success_rate
    }

if __name__ == '__main__':
    results = test_moe_abstract()
    print(f"\n✅ MoE Abstract Results:")
    print(f"Training: {results['training_success_rate']:.1%}")
    print(f"Novel: {results['novel_success_rate']:.1%}")


