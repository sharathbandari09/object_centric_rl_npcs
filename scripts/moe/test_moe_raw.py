#!/usr/bin/env python3
"""
Test the improved MoE model
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
from agents.moe.moe_raw import MoERaw

def test_moe_raw():
    """Test MoE Raw model"""
    print("=== TESTING MoE RAW ===")
    
    device = 'cpu'
    
    # Load model
    model = MoERaw(n_experts=4, input_dim=64, output_dim=6, device=device)
    checkpoint = torch.load(os.path.join(project_root, 'experiments/exp_moe_raw_keydoor_seed0/moe_raw_final.pth'), map_location=device)
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
                # Convert to pixel observation - handle variable grid sizes
                grid = obs['grid']
                grid_size = grid.shape[0]
                max_size = 12
                pixel_obs = np.zeros((1, max_size, max_size), dtype=np.float32)
                
                # Copy the actual grid data to the top-left corner
                for r in range(grid_size):
                    for c in range(grid_size):
                        cell_type = grid[r, c]
                        if cell_type == 1:  # Wall
                            pixel_obs[0, r, c] = 1.0
                        elif cell_type == 2:  # Agent
                            pixel_obs[0, r, c] = 0.5
                        elif cell_type == 3:  # Key
                            pixel_obs[0, r, c] = 0.8
                        elif cell_type == 4:  # Door
                            pixel_obs[0, r, c] = 0.3
                        # Floor (0) remains 0.0
                
                pixel_obs = pixel_obs.reshape(1, 1, max_size, max_size)
                
                # Get prediction
                with torch.no_grad():
                    pixel_tensor = torch.tensor(pixel_obs, dtype=torch.float32).to(device)
                    logits, gate_weights = model(pixel_tensor)
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
                # Convert to pixel observation - handle variable grid sizes
                grid = obs['grid']
                grid_size = grid.shape[0]
                max_size = 12
                pixel_obs = np.zeros((1, max_size, max_size), dtype=np.float32)
                
                # Copy the actual grid data to the top-left corner
                for r in range(grid_size):
                    for c in range(grid_size):
                        cell_type = grid[r, c]
                        if cell_type == 1:  # Wall
                            pixel_obs[0, r, c] = 1.0
                        elif cell_type == 2:  # Agent
                            pixel_obs[0, r, c] = 0.5
                        elif cell_type == 3:  # Key
                            pixel_obs[0, r, c] = 0.8
                        elif cell_type == 4:  # Door
                            pixel_obs[0, r, c] = 0.3
                        # Floor (0) remains 0.0
                
                pixel_obs = pixel_obs.reshape(1, 1, max_size, max_size)
                
                # Get prediction
                with torch.no_grad():
                    pixel_tensor = torch.tensor(pixel_obs, dtype=torch.float32).to(device)
                    logits, gate_weights = model(pixel_tensor)
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
    results = test_moe_raw()
    print(f"\n✅ MoE Raw Results:")
    print(f"Training: {results['training_success_rate']:.1%}")
    print(f"Novel: {results['novel_success_rate']:.1%}")
