#!/usr/bin/env python3
"""
Create dataset for the new KeyDoor environment with variable grid sizes.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pickle
from envs.keydoor.keydoor_env import KeyDoorEnv
import random

def get_oracle_action(obs, env):
    """Oracle policy: go to nearest key, then door"""
    agent_pos = obs['agent_pos']
    keys_collected = obs['keys_collected'][0]
    total_keys = obs['total_keys'][0]
    
    # If at key position, interact
    key_at_agent_pos = False
    for key_idx, key_pos in enumerate(env.current_key_positions):
        if np.allclose(agent_pos, key_pos) and key_pos[0] != -1:  # -1 means key was collected
            key_at_agent_pos = True
            break
    
    if key_at_agent_pos:
        return env.INTERACT
    
    # If at door position and all keys collected, interact
    if (np.allclose(agent_pos, env.current_door_position) and 
        keys_collected == total_keys and not env.door_open):
        return env.INTERACT
    
    # If all keys collected, go to door
    if keys_collected == total_keys:
        return get_direction_to_target(agent_pos, env.current_door_position, env)
    
    # Otherwise, go to nearest uncollected key
    nearest_key_pos = None
    min_distance = float('inf')
    
    for key_pos in env.current_key_positions:
        # Only consider keys that are still available (not collected)
        if key_pos[0] != -1:  # -1 means key was collected
            distance = abs(agent_pos[0] - key_pos[0]) + abs(agent_pos[1] - key_pos[1])
            if distance < min_distance:
                min_distance = distance
                nearest_key_pos = key_pos
    
    if nearest_key_pos is not None:
        return get_direction_to_target(agent_pos, nearest_key_pos, env)
    
    # Default: move randomly (should ideally not be reached if policy is good)
    return np.random.randint(0, 4)

def get_direction_to_target(agent_pos, target_pos, env):
    """Get direction to move towards target, avoiding walls"""
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    
    possible_moves = []
    if dx < 0: possible_moves.append(env.UP)
    if dx > 0: possible_moves.append(env.DOWN)
    if dy < 0: possible_moves.append(env.LEFT)
    if dy > 0: possible_moves.append(env.RIGHT)
    
    # Prioritize moves that reduce Manhattan distance
    best_action = -1
    min_dist_after_move = float('inf')
    
    for action in possible_moves:
        temp_pos = agent_pos.copy()
        if action == env.UP: temp_pos[0] -= 1
        elif action == env.DOWN: temp_pos[0] += 1
        elif action == env.LEFT: temp_pos[1] -= 1
        elif action == env.RIGHT: temp_pos[1] += 1
        
        if env._is_valid_position(temp_pos):
            dist_after_move = abs(temp_pos[0] - target_pos[0]) + abs(temp_pos[1] - target_pos[1])
            if dist_after_move < min_dist_after_move:
                min_dist_after_move = dist_after_move
                best_action = action
    
    if best_action != -1:
        return best_action
    
    # If no direct path, try any valid move
    valid_moves = []
    for action in range(4): # UP, DOWN, LEFT, RIGHT
        temp_pos = agent_pos.copy()
        if action == env.UP: temp_pos[0] -= 1
        elif action == env.DOWN: temp_pos[0] += 1
        elif action == env.LEFT: temp_pos[1] -= 1
        elif action == env.RIGHT: temp_pos[1] += 1
        
        if env._is_valid_position(temp_pos):
            valid_moves.append(action)
    
    if valid_moves:
        return random.choice(valid_moves)
    
    return env.NO_OP # If stuck

def generate_dataset(env, num_episodes_per_template=100):
    """Generate dataset using oracle policy"""
    raw_observations = []
    raw_actions = []
    entity_observations = []
    entity_actions = []
    
    success_count = 0
    total_episodes = 0
    
    for template_id in range(1, 7):  # Only training templates T1-T6
        print(f"Generating data for Template {template_id}...")
        
        for episode in range(num_episodes_per_template):
            obs = env.reset(template_id=template_id, seed=episode)
            done = False
            episode_obs = []
            episode_actions = []
            
            while not done:
                action = get_oracle_action(obs, env)
                episode_obs.append(obs)
                episode_actions.append(action)
                
                obs, reward, done, info = env.step(action)
            
            # Count success after episode is complete
            if info.get('success', False):
                success_count += 1
            total_episodes += 1
            
            # Add episode data
            raw_observations.extend(episode_obs)
            raw_actions.extend(episode_actions)
            entity_observations.extend(episode_obs)
            entity_actions.extend(episode_actions)
    
    success_rate = success_count / total_episodes if total_episodes > 0 else 0
    print(f"Oracle success rate: {success_rate:.2%} ({success_count}/{total_episodes})")
    
    return raw_observations, raw_actions, entity_observations, entity_actions

def main():
    """Generate and save datasets"""
    print("Creating new KeyDoor environment...")
    env = KeyDoorEnv()
    
    print("Generating dataset...")
    raw_obs, raw_actions, entity_obs, entity_actions = generate_dataset(env, num_episodes_per_template=50)
    
    print(f"Generated {len(raw_obs)} observations")
    
    # Save raw dataset
    raw_dataset = {
        'observations': raw_obs,
        'actions': raw_actions
    }
    
    with open('datasets/raw_keydoor_dataset_new.pkl', 'wb') as f:
        pickle.dump(raw_dataset, f)
    
    print("Raw dataset saved to datasets/raw_keydoor_dataset_new.pkl")
    
    # Save entity dataset
    entity_dataset = {
        'observations': entity_obs,
        'actions': entity_actions
    }
    
    with open('datasets/entity_keydoor_dataset_new.pkl', 'wb') as f:
        pickle.dump(entity_dataset, f)
    
    print("Entity dataset saved to datasets/entity_keydoor_dataset_new.pkl")
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total episodes: {len(raw_obs) // 50}")  # Assuming ~50 steps per episode
    print(f"Total observations: {len(raw_obs)}")
    print(f"Action distribution: {np.bincount(raw_actions, minlength=6)}")

if __name__ == "__main__":
    main()
