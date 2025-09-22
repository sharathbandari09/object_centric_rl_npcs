

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import random
from collections import deque
from pathlib import Path

class KeyDoorEnv(gym.Env):
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
        self.agent_pos = None
        self.keys_collected_count = 0
        self.total_keys_in_template = 0
        self.door_open = False
        self.step_count = 0
        self.seed = None

        # For rendering
        if self.render_mode == "human":
            import pygame
            pygame.init()
            self.cell_size = 60
            self.window_size = self.obs_grid_size * self.cell_size
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("KeyDoor Env")
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

        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        reward = -0.01  # Step penalty
        done = False
        info = {'template_id': self.current_template, 'seed': self.seed}

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
        elif action == self.INTERACT:
            reward += self._interact()
        elif action == self.NO_OP:
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check for goal
        if self._is_at_goal():
            reward += 1.0
            done = True
            info['success'] = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            info['success'] = False

        return self._get_observation(), reward, done, info

    def _interact(self) -> float:
        reward = 0.0

        # Check for key pickup
        for i, key_pos in enumerate(self.current_key_positions):
            if np.allclose(self.agent_pos, key_pos) and key_pos[0] != -1:  # -1 means key was collected
                self.keys_collected_count += 1
                self.grid[key_pos[0], key_pos[1]] = self.FLOOR
                reward += 0.1
                # Remove key from current_key_positions so it's not picked again
                self.current_key_positions[i] = np.array([-1, -1]) # Mark as collected
                break # Only pick one key per interact action

        # Check for door opening (only if all keys collected)
        if (self.keys_collected_count == self.total_keys_in_template and
            np.allclose(self.agent_pos, self.current_door_position) and
            not self.door_open):
            self.door_open = True
            self.grid[self.current_door_position[0], self.current_door_position[1]] = self.FLOOR
            reward += 0.1

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

        # Relative positions to agent
        features[3] = row - self.agent_pos[0]  # dx_agent
        features[4] = col - self.agent_pos[1]  # dy_agent

        # Global state
        features[5] = self.keys_collected_count # keys_collected
        features[6] = self.total_keys_in_template # total_keys
        features[7] = int(self.door_open) # door_open

        # Absolute positions (help model with spatial grounding)
        features[8] = row
        features[9] = col

        return features

    def _create_agent_entity_vector(self) -> np.ndarray:
        # Features: [type_onehot(3), dx_agent, dy_agent, keys_collected, total_keys, door_open, abs_row, abs_col]
        features = np.zeros(10)
        features[2] = 1.0 # Agent type

        # Relative positions to agent (0,0)
        features[3] = 0.0
        features[4] = 0.0

        # Global state
        features[5] = self.keys_collected_count
        features[6] = self.total_keys_in_template
        features[7] = int(self.door_open)

        # Absolute position of agent
        features[8] = self.agent_pos[0]
        features[9] = self.agent_pos[1]

        return features

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()

    def _render_pygame(self):
        import pygame
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("KeyDoor Env")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background

        for r in range(self.obs_grid_size):
            for c in range(self.obs_grid_size):
                tile_type = self.grid[r, c]
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                if tile_type == self.WALL:
                    pygame.draw.rect(canvas, (0, 0, 0), rect)  # Black wall
                elif tile_type == self.FLOOR:
                    pygame.draw.rect(canvas, (200, 200, 200), rect) # Light gray floor

        # Draw keys
        for key_pos in self.current_template['keys_pos']:
            if self.grid[key_pos[0], key_pos[1]] == self.KEY:
                pygame.draw.circle(canvas, (255, 255, 0), (key_pos[1] * self.cell_size + self.cell_size // 2, key_pos[0] * self.cell_size + self.cell_size // 2), self.cell_size // 3) # Yellow key

        # Draw door
        door_pos = self.current_template['door_pos']
        if self.grid[door_pos[0], door_pos[1]] == self.DOOR:
            pygame.draw.rect(canvas, (139, 69, 19), (door_pos[1] * self.cell_size, door_pos[0] * self.cell_size, self.cell_size, self.cell_size)) # Brown door

        # Draw agent
        pygame.draw.circle(canvas, (0, 0, 255), (self.agent_pos[1] * self.cell_size + self.cell_size // 2, self.agent_pos[0] * self.cell_size + self.cell_size // 2), self.cell_size // 3) # Blue agent

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if hasattr(self, 'window') and self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

    def is_solvable(self) -> bool:
        """Check if the current template is solvable using BFS pathfinding"""
        return self._check_solvability(self.current_template)

    def _check_solvability(self, template: Dict[str, Any]) -> bool:
        """Check if a template is solvable using BFS"""
        grid_size = template['grid_size']
        grid = template['grid']
        start_pos = template['agent_start']
        keys_pos = template['keys_pos']
        door_pos = template['door_pos']

        # Check if all keys are reachable from start
        for key_pos in keys_pos:
            if not self._bfs_path_exists(grid, start_pos, key_pos, grid_size):
                return False

        # Check if door is reachable from start (after collecting all keys)
        if not self._bfs_path_exists(grid, start_pos, door_pos, grid_size):
            return False

        return True

    def _bfs_path_exists(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], grid_size: int) -> bool:
        """Check if there's a path from start to goal using BFS"""
        if start == goal:
            return True

        visited = set()
        queue = deque([start])
        visited.add(start)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

        while queue:
            current = queue.popleft()
            
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (0 <= next_pos[0] < grid_size and 
                    0 <= next_pos[1] < grid_size and 
                    next_pos not in visited and
                    grid[next_pos[0], next_pos[1]] != self.WALL):
                    
                    if next_pos == goal:
                        return True
                    
                    visited.add(next_pos)
                    queue.append(next_pos)

        return False

