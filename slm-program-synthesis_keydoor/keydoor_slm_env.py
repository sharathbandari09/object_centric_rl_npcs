"""
KeyDoor SLM Environment - Optimized for Small Language Model Control
Designed specifically for SLM-based program synthesis and navigation planning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import random
import json
import os
from pathlib import Path

class KeyDoorSLMEnv(gym.Env):
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
        self.current_template_id = None
        self.seed = None
        
        # Environment state
        self.grid = None
        self.agent_pos = None
        self.current_key_positions = []
        self.current_door_position = None
        self.keys_collected = 0
        self.total_keys = 0
        self.door_open = False
        self.step_count = 0
        
        # SLM-specific state
        self.navigation_tasks = []
        self.current_task_index = 0
        self.completed_tasks = []
        self.spatial_blueprint = None

    def _load_templates(self) -> Dict[int, Dict[str, Any]]:
        """Load templates from JSON file"""
        # Get the project root directory (parent of slm-program-synthesis folder)
        project_root = Path(__file__).parent.parent
        template_path = project_root / "templates" / "keydoor_templates.json"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        with open(template_path, 'r') as f:
            templates_data = json.load(f)
        
        return templates_data

    def reset(self, template_id: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)

        # Select template
        if template_id is None:
            template_id = random.choice(list(self.templates.keys()))
        
        self.current_template_id = template_id
        self.current_template = self.templates[str(template_id)]
        
        # Initialize environment state
        self._initialize_environment()
        
        # Initialize SLM-specific state
        self._initialize_slm_state()
        
        self.step_count = 0
        
        return self._get_observation(), {}

    def _initialize_environment(self):
        """Initialize the environment grid and state from template."""
        grid_size = self.current_template['grid_size']
        grid_data = self.current_template['grid']
        
        # Create grid
        self.grid = np.zeros((self.obs_grid_size, self.obs_grid_size), dtype=np.int32)
        
        # Copy template grid to observation grid (centered)
        start_row = (self.obs_grid_size - grid_size) // 2
        start_col = (self.obs_grid_size - grid_size) // 2
        
        for row in range(grid_size):
            for col in range(grid_size):
                if row < len(grid_data) and col < len(grid_data[row]):
                    cell_value = grid_data[row][col]
                    obs_row = start_row + row
                    obs_col = start_col + col
                    
                    if cell_value == 1:  # Wall
                        self.grid[obs_row, obs_col] = self.WALL
                    elif cell_value == 2:  # Key
                        self.grid[obs_row, obs_col] = self.KEY
                    elif cell_value == 3:  # Door
                        self.grid[obs_row, obs_col] = self.DOOR
                    elif cell_value == 4:  # Agent start
                        self.grid[obs_row, obs_col] = self.FLOOR
                        self.agent_pos = np.array([obs_row, obs_col])
        
        # Set agent position from template (always use template, not grid cell 4)
        agent_start = self.current_template['agent_start']
        self.agent_pos = np.array([start_row + agent_start[0], start_col + agent_start[1]])
        
        # Initialize key and door positions
        self.current_key_positions = []
        for key_pos in self.current_template['keys_pos']:
            obs_key_pos = [start_row + key_pos[0], start_col + key_pos[1]]
            self.current_key_positions.append(obs_key_pos)
        
        door_pos = self.current_template['door_pos']  # door_pos is [row, col]
        self.current_door_position = np.array([start_row + door_pos[0], start_col + door_pos[1]])
        
        self.keys_collected = 0
        self.total_keys = len(self.current_key_positions)
        self.door_open = False

    def _initialize_slm_state(self):
        """Initialize SLM-specific state and navigation tasks."""
        # Create navigation tasks
        self.navigation_tasks = []
        self.current_task_index = 0
        self.completed_tasks = []
        
        # Get agent starting position
        agent_start = self.current_template['agent_start']
        
        # Add key collection tasks sorted by distance from agent
        key_tasks = []
        for i, key_pos in enumerate(self.current_template['keys_pos']):
            # Calculate Manhattan distance from agent to key
            distance = abs(key_pos[0] - agent_start[0]) + abs(key_pos[1] - agent_start[1])
            key_tasks.append({
                'task_id': f"key_{i+1}",
                'task_type': 'key',
                'target_position': tuple(key_pos),
                'completed': False,
                'distance': distance
            })
        
        # Sort keys by distance (nearest first)
        key_tasks.sort(key=lambda x: x['distance'])
        
        # Add sorted key tasks
        self.navigation_tasks.extend(key_tasks)
        
        # Add door opening task (use template coordinates)
        door_pos = self.current_template['door_pos']
        self.navigation_tasks.append({
            'task_id': 'door',
            'task_type': 'door',
            'target_position': tuple(door_pos),
            'completed': False
        })
        
        # Generate spatial blueprint for SLM
        self.spatial_blueprint = self._generate_spatial_blueprint()

    def _generate_spatial_blueprint(self) -> str:
        """Generate a spatial blueprint description for the SLM."""
        if not self.current_template:
            return "No template loaded"
        
        # Get current task
        current_task = self.get_current_navigation_task()
        if not current_task:
            return "All tasks completed"
        
        # Create map representation
        map_lines = []
        grid_size = self.current_template['grid_size']
        grid_data = self.current_template['grid']
        
        # Create ASCII map
        for row in range(grid_size):
            map_line = ""
            for col in range(grid_size):
                if row < len(grid_data) and col < len(grid_data[row]):
                    cell_value = grid_data[row][col]
                    if cell_value == 1:  # Wall
                        map_line += "#"
                    elif cell_value == 2:  # Key
                        map_line += "K"
                    elif cell_value == 3:  # Door
                        map_line += "D"
                    elif cell_value == 4:  # Agent start
                        map_line += "A"
                    else:  # Empty
                        map_line += "."
                else:
                    map_line += "."
            map_lines.append(map_line)
        
        # Add agent position marker
        agent_start = self.current_template['agent_start']
        if (agent_start[0] < len(map_lines) and 
            agent_start[1] < len(map_lines[agent_start[0]])):
            map_lines[agent_start[0]] = (map_lines[agent_start[0]][:agent_start[1]] + 
                                       "A" + 
                                       map_lines[agent_start[0]][agent_start[1]+1:])
        
        # Create blueprint description with template coordinates
        agent_template_pos = self.current_template['agent_start']
        blueprint = f"""KeyDoor Puzzle - Template {self.current_template_id}:
Grid Size: {grid_size}x{grid_size}
Agent Position: {agent_template_pos}
Current Task: {current_task['task_type'].upper()} at {current_task['target_position']}
Keys Collected: {self.keys_collected}/{self.total_keys}
Door Status: {'OPEN' if self.door_open else 'CLOSED'}

Map Blueprint:
{chr(10).join(map_lines)}

Legend:
# = Wall (impassable)
. = Empty space
K = Key (collectible)
D = Door (openable after collecting all keys)
A = Agent (current position)

Objective: Navigate to collect all keys, then open the door."""
        
        return blueprint

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # SLM environments don't need complex reward structures
        # Success is determined by task completion, not reward maximization
        reward = 0.0
        done = False
        info = {
            'template_id': self.current_template_id, 
            'seed': self.seed,
            'current_task': self.get_current_navigation_task(),
            'spatial_blueprint': self.spatial_blueprint
        }

        new_pos = self.agent_pos.copy()
        if action == self.UP:
            new_pos[0] -= 1
        elif action == self.DOWN:
            new_pos[0] += 1
        elif action == self.LEFT:
            new_pos[1] -= 1
        elif action == self.RIGHT:
            new_pos[1] += 1
        elif action == self.INTERACT:
            self._interact()
        elif action == self.NO_OP:
            pass

        # Check if movement is valid
        if action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
            # No penalty for invalid moves - SLM should learn to avoid them

        self.step_count += 1

        # Check if current task is completed
        if self._is_current_task_completed():
            self._complete_current_task()
            # Regenerate spatial blueprint for next task
            self.spatial_blueprint = self._generate_spatial_blueprint()

        # Check if all tasks are completed
        if self._are_all_tasks_completed():
            done = True
            reward = 1.0  # Success reward

        # Check for timeout
        if self.step_count >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, info

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is valid (within bounds and not a wall)."""
        row, col = pos
        if (row < 0 or row >= self.obs_grid_size or 
            col < 0 or col >= self.obs_grid_size):
            return False
        return self.grid[row, col] != self.WALL

    def _interact(self):
        """Handle interaction with keys and doors."""
        agent_pos_tuple = tuple(self.agent_pos)
        
        # Check for key collection
        if agent_pos_tuple in [tuple(pos) for pos in self.current_key_positions]:
            # Remove key from grid and list
            self.grid[agent_pos_tuple[0], agent_pos_tuple[1]] = self.FLOOR
            self.current_key_positions = [pos for pos in self.current_key_positions 
                                        if tuple(pos) != agent_pos_tuple]
            self.keys_collected += 1
        
        # Check for door opening
        elif (agent_pos_tuple == tuple(self.current_door_position) and 
              self.keys_collected == self.total_keys and 
              not self.door_open):
            self.grid[agent_pos_tuple[0], agent_pos_tuple[1]] = self.FLOOR
            self.door_open = True

    def _is_current_task_completed(self) -> bool:
        """Check if the current navigation task is completed."""
        current_task = self.get_current_navigation_task()
        if not current_task:
            return True
        
        # Convert agent position to template coordinates
        start_row = (self.obs_grid_size - self.current_template['grid_size']) // 2
        start_col = (self.obs_grid_size - self.current_template['grid_size']) // 2
        agent_template_pos = (self.agent_pos[0] - start_row, self.agent_pos[1] - start_col)
        
        if current_task['task_type'] == 'key':
            return agent_template_pos == current_task['target_position']
        elif current_task['task_type'] == 'door':
            return (agent_template_pos == current_task['target_position'] and 
                    self.keys_collected == self.total_keys and 
                    self.door_open)
        
        return False

    def _complete_current_task(self):
        """Mark current task as completed and move to next."""
        if self.current_task_index < len(self.navigation_tasks):
            current_task = self.navigation_tasks[self.current_task_index]
            current_task['completed'] = True
            self.completed_tasks.append(current_task)
            self.current_task_index += 1

    def _are_all_tasks_completed(self) -> bool:
        """Check if all navigation tasks are completed."""
        return self.current_task_index >= len(self.navigation_tasks)

    def get_current_navigation_task(self) -> Optional[Dict[str, Any]]:
        """Get the current navigation task."""
        if self.current_task_index < len(self.navigation_tasks):
            return self.navigation_tasks[self.current_task_index]
        return None

    def get_spatial_blueprint(self) -> str:
        """Get the current spatial blueprint for SLM."""
        return self.spatial_blueprint

    def get_navigation_context(self) -> Dict[str, Any]:
        """Get navigation context for SLM planning."""
        # Reorder remaining key tasks by current distance
        self._reorder_remaining_keys_by_distance()
        
        current_task = self.get_current_navigation_task()
        if not current_task:
            return {"error": "All tasks completed"}
        
        # Convert observation coordinates back to template coordinates
        start_row = (self.obs_grid_size - self.current_template['grid_size']) // 2
        start_col = (self.obs_grid_size - self.current_template['grid_size']) // 2
        
        agent_template_pos = (self.agent_pos[0] - start_row, self.agent_pos[1] - start_col)
        
        return {
            "agent_position": agent_template_pos,
            "current_task": current_task['task_type'],
            "target_position": current_task['target_position'],
            "target_id": current_task['task_id'],
            "keys_collected": self.keys_collected,
            "total_keys": self.total_keys,
            "door_open": self.door_open,
            "spatial_blueprint": self.spatial_blueprint,
            "remaining_tasks": len(self.navigation_tasks) - self.current_task_index
        }
    
    def _reorder_remaining_keys_by_distance(self):
        """Reorder remaining key tasks by distance from current agent position."""
        # Convert observation coordinates back to template coordinates
        start_row = (self.obs_grid_size - self.current_template['grid_size']) // 2
        start_col = (self.obs_grid_size - self.current_template['grid_size']) // 2
        agent_template_pos = (self.agent_pos[0] - start_row, self.agent_pos[1] - start_col)
        
        # Find remaining key tasks
        remaining_key_tasks = []
        door_task = None
        
        for i in range(self.current_task_index, len(self.navigation_tasks)):
            task = self.navigation_tasks[i]
            if task['task_type'] == 'key' and not task['completed']:
                # Calculate current distance from agent to key
                distance = abs(task['target_position'][0] - agent_template_pos[0]) + abs(task['target_position'][1] - agent_template_pos[1])
                task['distance'] = distance
                remaining_key_tasks.append(task)
            elif task['task_type'] == 'door':
                door_task = task
        
        # Sort remaining keys by current distance
        remaining_key_tasks.sort(key=lambda x: x['distance'])
        
        # Rebuild navigation tasks with reordered keys
        new_tasks = []
        
        # Keep completed tasks
        for i in range(self.current_task_index):
            new_tasks.append(self.navigation_tasks[i])
        
        # Add reordered remaining key tasks
        new_tasks.extend(remaining_key_tasks)
        
        # Add door task at the end
        if door_task:
            new_tasks.append(door_task)
        
        # Update navigation tasks
        self.navigation_tasks = new_tasks

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation."""
        # Create observation grid with agent position
        obs_grid = self.grid.copy()
        obs_grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT
        
        # Create entities list
        entities = np.zeros((16, 10), dtype=np.float32)
        entity_mask = np.zeros(16, dtype=np.bool_)
        
        entity_idx = 0
        for key_pos in self.current_key_positions:
            if entity_idx < 16:
                # Entity features: [type_onehot(3), dx, dy, keys_col, total_keys, door_open, abs_row, abs_col]
                entities[entity_idx] = [
                    0, 0, 1,  # Key type onehot
                    key_pos[0] - self.agent_pos[0],  # dx
                    key_pos[1] - self.agent_pos[1],  # dy
                    self.keys_collected,  # keys_collected
                    self.total_keys,  # total_keys
                    1 if self.door_open else 0,  # door_open
                    key_pos[0],  # abs_row
                    key_pos[1]   # abs_col
                ]
                entity_mask[entity_idx] = True
                entity_idx += 1
        
        # Add door entity
        if entity_idx < 16:
            entities[entity_idx] = [
                0, 1, 0,  # Door type onehot
                self.current_door_position[0] - self.agent_pos[0],  # dx
                self.current_door_position[1] - self.agent_pos[1],  # dy
                self.keys_collected,  # keys_collected
                self.total_keys,  # total_keys
                1 if self.door_open else 0,  # door_open
                self.current_door_position[0],  # abs_row
                self.current_door_position[1]   # abs_col
            ]
            entity_mask[entity_idx] = True
        
        return {
            "grid": obs_grid,
            "agent_pos": self.agent_pos.astype(np.int32),
            "keys_collected": np.array([self.keys_collected], dtype=np.int32),
            "total_keys": np.array([self.total_keys], dtype=np.int32),
            "door_open": np.array([1 if self.door_open else 0], dtype=np.int32),
            "entities": entities,
            "entity_mask": entity_mask,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.step_count}")
            print(f"Agent: {tuple(self.agent_pos)}")
            print(f"Keys: {self.keys_collected}/{self.total_keys}")
            print(f"Door: {'OPEN' if self.door_open else 'CLOSED'}")
            print(f"Current Task: {self.get_current_navigation_task()}")
            print("Grid:")
            for row in self.grid:
                print("".join(str(cell) for cell in row))
            print()

    def close(self):
        """Close the environment."""
        pass
