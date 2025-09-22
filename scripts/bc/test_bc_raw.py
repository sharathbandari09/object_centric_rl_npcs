#!/usr/bin/env python3
"""
Test BC Raw model on training and novel layouts
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

class BCRaw(torch.nn.Module):
    def __init__(self, input_channels=1, max_grid_size=12, n_actions=6, hidden_dim=128):
        super().__init__()
        # Simple CNN for pixel input
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4))  # Reduce to 4x4
        )
        
        # Calculate flattened size
        self.flattened_size = 64 * 4 * 4
        
        # Policy head
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(self.flattened_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, pixel_obs):
        # pixel_obs shape: (batch_size, 1, grid_size, grid_size)
        conv_out = self.conv_layers(pixel_obs)
        flattened = conv_out.view(conv_out.size(0), -1)
        action_logits = self.policy_head(flattened)
        return action_logits

def test_bc_raw():
    print("=== Testing BC Raw Model ===")
    
    device = 'cpu'
    
    # Load model
    model = BCRaw().to(device)
    checkpoint = torch.load(os.path.join(project_root, 'experiments/exp_bc_raw_keydoor_seed0/bc_raw_final.pth'), map_location=device)
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
                # Convert grid to pixel observation - handle variable grid sizes
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
                
                pixel_obs = pixel_obs.reshape(1, 1, max_size, max_size)  # (1, 1, 12, 12)
                
                # Get model prediction
                with torch.no_grad():
                    pixel_tensor = torch.tensor(pixel_obs, dtype=torch.float32).to(device)
                    logits = model(pixel_tensor)
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
                # Convert grid to pixel observation - handle variable grid sizes
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
                
                pixel_obs = pixel_obs.reshape(1, 1, max_size, max_size)  # (1, 1, 12, 12)
                
                # Get model prediction
                with torch.no_grad():
                    pixel_tensor = torch.tensor(pixel_obs, dtype=torch.float32).to(device)
                    logits = model(pixel_tensor)
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
    
    print(f"\n=== BC RAW RESULTS ===")
    print(f"Training Layouts (T1-T6): {training_rate:.2f} ({training_success}/{training_total})")
    print(f"Novel Layouts (T7-T10): {novel_rate:.2f} ({novel_success}/{novel_total})")
    print(f"Expected: Limited generalization")
    print(f"Actual: {training_rate:.0%} on training, {novel_rate:.0%} on novel")
    
    env.close()
    return training_rate, novel_rate

if __name__ == '__main__':
    test_bc_raw()
