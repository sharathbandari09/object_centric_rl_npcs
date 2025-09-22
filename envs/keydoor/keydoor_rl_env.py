"""
KeyDoor RL Environment - Optimized for Reinforcement Learning
Same gameplay as KeyDoor but with dense rewards and progress signals
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import random
from collections import deque
from pathlib import Path

class KeyDoorRLEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Define constants for grid elements
    FLOOR = 0
    WALL = 1
    AGENT = 2
    KEY = 3
    DOOR = 4

    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4
    NO_OP = 5

    def __init__(self, grid_size=8, max_steps=200, render_mode: Optional[str] = None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Use fixed observation space for compatibility (pad to 12x12)
        self.obs_grid_size = 12
        self.action_space = spaces.Discrete(6)  # UP, DOWN, LEFT, RIGHT, INTERACT, NO_OP
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 4, (self.obs_grid_size, self.obs_grid_size), dtype=np.int32),
            "agent_pos": spaces.Box(0, self.obs_grid_size - 1, (2,), dtype=np.int32),
            "keys_collected": spaces.Box(0, 10, (1,), dtype=np.int32), # Max 10 keys
            "total_keys": spaces.Box(0, 10, (1,), dtype=np.int32),
            "door_open": spaces.Box(0, 1, (1,), dtype=np.int32),
            # Max 16 entities, 10 features (type_onehot(3), dx, dy, keys_col, total_keys, door_open, abs_row, abs_col)
            "entities": spaces.Box(low=-np.inf, high=np.inf, shape=(16, 10), dtype=np.float32),
            "entity_mask": spaces.Box(0, 1, (16,), dtype=np.bool_),
        })

        self.templates = self._load_templates()
        self.current_template = None
        
        # RL-specific tracking
        self.prev_agent_pos = None
        self.prev_key_distances = None
        self.prev_door_distance = None
        self.visited_positions = set()
        self.stuck_counter = 0
        self.last_action = None
        self.action_repeat_counter = 0

        # For rendering
        if self.render_mode == "human":
            import pygame
            pygame.init()
            self.cell_size = 60
            self.window_size = self.obs_grid_size * self.cell_size
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("KeyDoor RL Env")
            self.clock = pygame.time.Clock()

    def _load_templates(self) -> Dict[int, Dict[str, Any]]:
        """Load templates from JSON file"""
        import json
        import os
        
        # Get the project root directory (parent of envs folder)
        project_root = Path(__file__).parent.parent.parent
        template_file = project_root / "templates" / "keydoor_templates.json"
        if not template_file.exists():
            raise FileNotFoundError(f"Template file {template_file} not found. Please run scripts/generate_templates.py first.")
        
        with open(template_file, 'r') as f:
            templates_data = json.load(f)
        
        # Convert to numpy arrays and tuples
        templates = {}
        for template_id, template_data in templates_data.items():
            template_id = int(template_id)
            templates[template_id] = {
                'grid_size': template_data['grid_size'],
                'grid': np.array(template_data['grid']),
                'agent_start': tuple(template_data['agent_start']),
                'keys_pos': [tuple(pos) for pos in template_data['keys_pos']],
                'door_pos': tuple(template_data['door_pos']),
                'template_id': template_id,
                'num_keys': template_data['num_keys']
            }
        return templates

    def reset(self, template_id: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)

        if template_id is None:
            template_id = np.random.randint(1, len(self.templates) + 1)
        self.current_template = self.templates[template_id]

        # Set grid size based on template
        self.grid_size = self.current_template['grid_size']
        
        # Initialize grid with padding to obs_grid_size
        self.grid = np.zeros((self.obs_grid_size, self.obs_grid_size), dtype=np.int32)
        self.grid[:self.grid_size, :self.grid_size] = self.current_template['grid']
        
        # Set agent position
        self.agent_pos = np.array(self.current_template['agent_start'])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

        self.keys_collected_count = 0
        self.total_keys_in_template = len(self.current_template['keys_pos'])
        self.door_open = False
        self.step_count = 0

        # Store original key positions for interaction logic
        self.current_key_positions = [np.array(pos) for pos in self.current_template['keys_pos']]
        self.current_door_position = np.array(self.current_template['door_pos'])

        # Reset RL-specific tracking
        self.prev_agent_pos = None
        self.prev_key_distances = None
        self.prev_door_distance = None
        self.visited_positions = set()
        self.stuck_counter = 0
        self.last_action = None
        self.action_repeat_counter = 0

        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # Base reward structure (much more generous for RL)
        reward = 0.0  # No step penalty
        done = False
        info = {'template_id': self.current_template, 'seed': self.seed}

        # Track action repetition
        if action == self.last_action:
            self.action_repeat_counter += 1
        else:
            self.action_repeat_counter = 0
        self.last_action = action

        new_pos = self.agent_pos.copy()
        if action == self.UP:
            new_pos[0] -= 1
        elif action == self.DOWN:
            new_pos[0] += 1
        elif action == self.LEFT:
            new_pos[1] -= 1
        elif action == self.RIGHT:
            new_pos[1] += 1

        if action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
            if self._is_valid_position(new_pos):
                self.grid[self.agent_pos[0], self.agent_pos[1]] = self.FLOOR
                self.agent_pos = new_pos
                self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT
                # Reward for successful movement
                reward += 0.01
            else:
                # Small penalty for hitting wall
                reward -= 0.05
        elif action == self.INTERACT:
            interaction_reward = self._interact()
            reward += interaction_reward
        elif action == self.NO_OP:
            # Small penalty for doing nothing
            reward -= 0.02

        # Calculate dense rewards
        dense_reward = self._calculate_dense_rewards()
        reward += dense_reward

        # Check for goal
        if self._is_at_goal():
            reward += 10.0  # Large success bonus
            done = True
            info['success'] = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            info['success'] = False

        return self._get_observation(), reward, done, info

    def _calculate_dense_rewards(self) -> float:
        """Calculate dense rewards to guide RL learning"""
        dense_reward = 0.0
        
        # Distance-based rewards
        if self.current_key_positions:
            # Distance to nearest uncollected key
            key_distances = []
            for key_pos in self.current_key_positions:
                if key_pos[0] != -1:  # Key not collected
                    dist = np.linalg.norm(self.agent_pos - key_pos)
                    key_distances.append(dist)
            
            if key_distances:
                min_key_distance = min(key_distances)
                
                # Reward for getting closer to nearest key
                if self.prev_key_distances is not None:
                    prev_min_key_distance = min(self.prev_key_distances) if self.prev_key_distances else float('inf')
                    if min_key_distance < prev_min_key_distance:
                        dense_reward += 0.1  # Getting closer to key
                    elif min_key_distance > prev_min_key_distance:
                        dense_reward -= 0.05  # Getting farther from key
                
                self.prev_key_distances = key_distances
        else:
            # All keys collected, reward getting closer to door
            door_distance = np.linalg.norm(self.agent_pos - self.current_door_position)
            
            if self.prev_door_distance is not None:
                if door_distance < self.prev_door_distance:
                    dense_reward += 0.1  # Getting closer to door
                elif door_distance > self.prev_door_distance:
                    dense_reward -= 0.05  # Getting farther from door
            
            self.prev_door_distance = door_distance
        
        # Exploration bonus
        agent_pos_tuple = tuple(self.agent_pos)
        if agent_pos_tuple not in self.visited_positions:
            dense_reward += 0.05
            self.visited_positions.add(agent_pos_tuple)
        
        # Stuck penalty
        if self.prev_agent_pos is not None and np.array_equal(self.agent_pos, self.prev_agent_pos):
            self.stuck_counter += 1
            if self.stuck_counter > 5:
                dense_reward -= 0.1  # Penalty for being stuck
        else:
            self.stuck_counter = 0
        
        # Action repetition penalty
        if self.action_repeat_counter > 3:
            dense_reward -= 0.1
        
        # Update previous position
        self.prev_agent_pos = self.agent_pos.copy()
        
        return dense_reward

    def _interact(self) -> float:
        reward = 0.0

        # Check for key pickup
        for i, key_pos in enumerate(self.current_key_positions):
            if np.allclose(self.agent_pos, key_pos) and key_pos[0] != -1:  # -1 means key was collected
                self.keys_collected_count += 1
                self.grid[key_pos[0], key_pos[1]] = self.FLOOR
                reward += 2.0  # Large reward for key collection
                # Remove key from current_key_positions so it's not picked again
                self.current_key_positions[i] = np.array([-1, -1]) # Mark as collected
                break # Only pick one key per interact action

        # Check for door opening (only if all keys collected)
        if (self.keys_collected_count == self.total_keys_in_template and
            np.allclose(self.agent_pos, self.current_door_position) and
            not self.door_open):
            self.door_open = True
            self.grid[self.current_door_position[0], self.current_door_position[1]] = self.FLOOR
            reward += 5.0  # Large reward for door opening

        return reward

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
            return False
        return self.grid[pos[0], pos[1]] != self.WALL

    def _is_at_goal(self) -> bool:
        """Check if agent has successfully interacted with the door (goal)."""
        return self.door_open

    def _get_observation(self) -> Dict[str, Any]:
        obs = {
            "grid": self.grid.copy(),
            "agent_pos": self.agent_pos.copy(),
            "keys_collected": np.array([self.keys_collected_count], dtype=np.int32),
            "total_keys": np.array([self.total_keys_in_template], dtype=np.int32),
            "door_open": np.array([int(self.door_open)], dtype=np.int32),
        }
        entities, entity_mask = self._extract_entities()
        obs["entities"] = entities
        obs["entity_mask"] = entity_mask
        return obs

    def _extract_entities(self) -> Tuple[np.ndarray, np.ndarray]:
        entities = []
        max_entities = 16 # Agent + keys + door

        # Agent entity
        entities.append(self._create_agent_entity_vector())

        # Key entities
        for key_pos in self.current_key_positions:
            if key_pos[0] != -1 and self.grid[key_pos[0], key_pos[1]] == self.KEY: # Only add if still in grid
                entities.append(self._create_entity_vector(self.KEY, key_pos[0], key_pos[1]))

        # Door entity (always include, even if opened)
        entities.append(self._create_entity_vector(self.DOOR, self.current_door_position[0], self.current_door_position[1]))

        # Pad or truncate
        entity_mask = np.zeros(max_entities, dtype=np.bool_)
        for i in range(min(len(entities), max_entities)):
            entity_mask[i] = True
        while len(entities) < max_entities:
            entities.append(np.zeros(10)) # 10 features per entity
        entities = entities[:max_entities]

        return np.array(entities, dtype=np.float32), entity_mask

    def _create_entity_vector(self, entity_type: int, row: int, col: int) -> np.ndarray:
        # Features: [type_onehot(3), dx_agent, dy_agent, keys_collected, total_keys, door_open, abs_row, abs_col]
        features = np.zeros(10)

        # Type one-hot (KEY=0, DOOR=1, AGENT=2)
        if entity_type == self.KEY:
            features[0] = 1.0
        elif entity_type == self.DOOR:
            features[1] = 1.0
        elif entity_type == self.AGENT:
            features[2] = 1.0

        # Relative position to agent
        features[3] = row - self.agent_pos[0]  # dx
        features[4] = col - self.agent_pos[1]  # dy

        # Game state
        features[5] = self.keys_collected_count
        features[6] = self.total_keys_in_template
        features[7] = int(self.door_open)

        # Absolute position
        features[8] = row
        features[9] = col

        return features

    def _create_agent_entity_vector(self) -> np.ndarray:
        return self._create_entity_vector(self.AGENT, self.agent_pos[0], self.agent_pos[1])

    def render(self):
        if self.render_mode == "human":
            import pygame
            
            # Clear screen
            self.window.fill((255, 255, 255))
            
            # Draw grid
            for i in range(self.obs_grid_size):
                for j in range(self.obs_grid_size):
                    if i < self.grid_size and j < self.grid_size:
                        cell_value = self.grid[i, j]
                        color = self._get_cell_color(cell_value)
                        pygame.draw.rect(self.window, color, 
                                       (j * self.cell_size, i * self.cell_size, 
                                        self.cell_size, self.cell_size))
                        pygame.draw.rect(self.window, (0, 0, 0), 
                                       (j * self.cell_size, i * self.cell_size, 
                                        self.cell_size, self.cell_size), 1)
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _get_cell_color(self, cell_value: int) -> Tuple[int, int, int]:
        colors = {
            self.FLOOR: (255, 255, 255),  # White
            self.WALL: (0, 0, 0),         # Black
            self.AGENT: (0, 255, 0),      # Green
            self.KEY: (255, 255, 0),      # Yellow
            self.DOOR: (255, 0, 0),       # Red
        }
        return colors.get(cell_value, (128, 128, 128))

    def close(self):
        if hasattr(self, 'window'):
            import pygame
            pygame.quit()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]







