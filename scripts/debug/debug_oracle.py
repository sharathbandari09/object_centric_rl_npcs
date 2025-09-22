# scripts/debug_oracle.py

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.boxpush.boxpush_env import BoxPushEnv

def print_grid(grid):
    """Print the grid in a readable format"""
    symbols = {0: '.', 1: '#', 2: 'A', 3: 'B', 4: 'T'}
    for row in grid:
        print(''.join(symbols.get(cell, '?') for cell in row))

def simple_oracle_action(obs):
    """Very simple oracle - just move towards nearest box"""
    agent_pos = tuple(obs.get("agent_pos", [0, 0]))
    grid = np.array(obs.get("grid", []))
    
    if grid.size == 0:
        return 5  # NO_OP
    
    # Find boxes
    boxes = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 3:  # BOX
                boxes.append((r, c))
    
    if not boxes:
        return 5  # NO_OP
    
    # Find nearest box
    nearest_box = min(boxes, key=lambda b: abs(b[0] - agent_pos[0]) + abs(b[1] - agent_pos[1]))
    
    # Simple movement towards nearest box
    dr = nearest_box[0] - agent_pos[0]
    dc = nearest_box[1] - agent_pos[1]
    
    if abs(dr) > abs(dc):
        return 1 if dr > 0 else 0  # DOWN if dr > 0, UP if dr < 0
    else:
        return 3 if dc > 0 else 2  # RIGHT if dc > 0, LEFT if dc < 0

def test_oracle_on_template(template_id=1, max_steps=20):
    """Test the oracle on a specific template"""
    env = BoxPushEnv()
    
    print(f"Testing Template {template_id}")
    print("=" * 40)
    
    obs, info = env.reset(template_id=template_id)
    
    print("Initial state:")
    print_grid(obs['grid'])
    print(f"Agent pos: {obs['agent_pos']}")
    print(f"Boxes pushed: {obs['boxes_pushed']}/{obs['total_boxes']}")
    print(f"Task complete: {obs['task_complete']}")
    print()
    
    for step in range(max_steps):
        action = simple_oracle_action(obs)
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PUSH", 5: "NO_OP"}
        
        print(f"Step {step + 1}: Action = {action_names[action]}")
        
        obs, reward, done, info = env.step(action)
        
        print(f"Reward: {reward}, Done: {done}")
        print(f"Agent pos: {obs['agent_pos']}")
        print(f"Boxes pushed: {obs['boxes_pushed']}/{obs['total_boxes']}")
        print(f"Task complete: {obs['task_complete']}")
        
        # Print grid
        print_grid(obs['grid'])
        print()
        
        if done:
            print(f"Episode finished! Success: {obs['task_complete']}")
            break
    
    return obs['task_complete'] == 1

if __name__ == "__main__":
    # Test on first few templates
    for template_id in range(1, 4):
        success = test_oracle_on_template(template_id)
        print(f"Template {template_id} success: {success}")
        print("=" * 60)
        print()
