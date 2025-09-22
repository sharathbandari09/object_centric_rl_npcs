"""
Simple Dungeon - Step 1: Basic Map Structure
Just a big horizontal map with outer borders and camera following player
"""

import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 32
GRID_WIDTH = 80  # Much wider horizontal map
GRID_HEIGHT = 20
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 650
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
PLAYER_BLUE = (0, 100, 255)
DARK_GRAY = (64, 64, 64)
PURPLE = (128, 0, 128)
GOLD = (255, 215, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
BROWN = (139, 69, 19)
DARK_RED = (139, 0, 0)

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0
    
    def update(self, target_x, target_y):
        # Center camera on target
        self.x = target_x * TILE_SIZE - self.width // 2
        self.y = target_y * TILE_SIZE - self.height // 2
        
        # Clamp to world bounds
        max_x = GRID_WIDTH * TILE_SIZE - self.width
        max_y = GRID_HEIGHT * TILE_SIZE - self.height
        
        self.x = max(0, min(self.x, max_x))
        self.y = max(0, min(self.y, max_y))

class SimpleDungeon:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Simple Dungeon - Step 1: Basic Map")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Camera
        self.camera = Camera(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Game state
        self.running = True
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        
        # Maze walls (for thin line rendering)
        self.maze_walls = {
            'horizontal': set(),  # (x, y) positions for horizontal walls
            'vertical': set()     # (x, y) positions for vertical walls
        }
        
        # Player starts in right chamber
        self.player_pos = [GRID_HEIGHT//2, GRID_WIDTH - 5]  # [y, x] - right chamber
        
        # Movement timing for continuous movement
        self.last_move_time = 0
        self.move_delay = 150  # milliseconds between continuous moves
        
        # Game state for box interactions
        self.box_positions = {}  # Track moveable box positions
        self.attached_box = None  # Box currently attached to player
        self.attachment_offset = None  # Relative position of attached box to player
        
        # Win condition and key system
        self.keys_released = False
        self.secret_keys = self._generate_secret_keys()  # Random 5-digit values
        self.slot_positions = []  # Will store slot positions
        self.key_objects = {}  # Will store visual key objects when spawned
        self.collected_keys = []  # Player's key inventory
        self.key_display_timer = 0  # Timer for showing key values
        self.key_display_duration = 3000  # Show for 3 seconds
        
        # Guard system
        self.guard_pos = None  # Will store guard position
        self.guard_challenge_active = False
        self.guard_passed = False  # Whether player has passed the guard challenge  
        self.guard_password_ui_active = False  # Password popup UI
        self.guard_password_input = ""  # Current password input
        self.guard_expected_passwords = []  # Valid passwords for this session
        self.guard_digit_challenges = []  # Specific digit challenges
        
        # Maze navigation system
        self.maze_map_spawned = False  # Whether navigation map has been spawned
        self.maze_map_collected = False  # Whether navigation map has been collected
        self.maze_map_pos = None  # Position of the navigation map item
        self.maze_solution_path = []  # Actual solution path through maze
        self.inverted_instructions = []  # Inverted navigation instructions
        self.map_ui_active = False  # Whether map reading UI is active
        
        # A* route visualization
        self.astar_path = []  # Store the optimal A* path
        self.show_astar_route = False  # Toggle for showing red route
        
        # Step tracking for debugging
        self.step_debug_active = False  # Whether to track movement steps
        self.last_position = None  # Last player position for direction calculation
        self.step_count = 0  # Number of steps taken
        self.player_in_maze = False  # Whether player is currently inside the maze
        
        # Master key challenge system
        self.master_key_pos = None  # Position of master key
        self.master_key_collected = False  # Whether master key has been collected
        self.master_key_challenge_active = False  # Whether master key challenge is active
        self.master_key_password_ui_active = False  # Password popup UI for master key
        self.master_key_password_input = ""  # Current password input for master key
        self.master_key_expected_password = None  # Expected 4-digit password
        self.master_key_operation = None  # Current math operation (add/multiply/divide)
        self.master_key_operation_text = ""  # Human-readable operation description
        
        # Queen release challenge system
        self.original_box_positions = {}  # Store original RGB box positions
        self.boxes_in_original_position = False  # Whether boxes are correctly arranged
        self.queen_challenge_active = False  # Whether queen challenge is active
        self.queen_password_ui_active = False  # Password popup UI for queen
        self.queen_password_input = ""  # Current password input for queen
        self.queen_expected_password = ""  # Expected queen password
        self.queen_attempts = 0  # Number of password attempts (max 3)
        self.queen_released = False  # Whether queen has been released
        
        # Info popup system
        self.info_popup_active = False  # Whether info popup is showing
        self.info_popup_title = ""  # Title of the info popup
        self.info_popup_messages = []  # List of messages to show in popup
        self.info_popup_auto_close_time = 0  # Time when popup should auto-close (0 = manual)
        self.info_popup_trigger_pos = None  # Position that triggers the popup
        self.info_popup_trigger_distance = 2  # Distance to trigger popup
        
        # Initialize map
        self._create_basic_map()
        
    def _create_basic_map(self):
        """Create chamber-based layout with rectangular rooms connected by passages."""
        # Fill with floor
        self.grid.fill(0)  # 0 = floor
        
        # Create basic outer walls
        self.grid[0, :] = 1  # Top wall
        self.grid[-1, :] = 1  # Bottom wall
        self.grid[:, 0] = 1  # Left wall
        self.grid[:, -1] = 1  # Right wall
        
        # Create entrance on left side
        entrance_y = GRID_HEIGHT // 2
        self.grid[entrance_y, 0] = 0  # Open entrance
        self.grid[entrance_y-1, 0] = 0  # Make entrance 2 tiles high
        
        # Define chamber boundaries
        left_chamber_end = 20
        middle_passage_start = 25
        middle_passage_end = 55
        right_chamber_start = 60
        
        # Create narrow middle passage (3 tiles high)
        passage_top = GRID_HEIGHT // 2 - 1
        passage_bottom = GRID_HEIGHT // 2 + 1
        
        # Walls above and below the narrow passage only
        for x in range(middle_passage_start, middle_passage_end):
            for y in range(1, passage_top):
                self.grid[y, x] = 1  # Top walls
            for y in range(passage_bottom + 1, GRID_HEIGHT - 1):
                self.grid[y, x] = 1  # Bottom walls
        
        # Generate left chamber maze
        self._generate_left_maze(left_chamber_end)
        
        # Add prison cell in right chamber with queen
        self._add_right_chamber_objects()
        
        # Add guard at entrance to left chamber
        self._add_guard_at_left_entrance(middle_passage_start)
        
        # Add master key in far left of left chamber
        self._add_master_key_in_left_chamber()
        
        print(f"Map created: {GRID_WIDTH}x{GRID_HEIGHT}")
        print(f"Left chamber: 0-{left_chamber_end}, Passage: {middle_passage_start}-{middle_passage_end}, Right chamber: {right_chamber_start}-{GRID_WIDTH}")
        print(f"Player starts at: {self.player_pos}")
        
    def _generate_left_maze(self, maze_end_x):
        """Generate complex random maze using algorithm from Random labyrinth.py."""
        # Define maze boundaries - properly centered and smaller
        available_width = maze_end_x - 2  # Leave 1 tile padding from outer walls
        available_height = GRID_HEIGHT - 4  # Leave 2 tiles padding from top/bottom
        
        maze_width = int(available_width * 0.7)  # 70% of available width
        maze_height = int(available_height * 0.7)  # 70% of available height
        
        # Center the maze in the left chamber, slightly right
        left_padding = (available_width - maze_width) // 2 + 3  # +4 for left wall offset + slight right shift
        top_padding = (available_height - maze_height) // 2 + 2  # +2 for top wall offset
        
        maze_left = int(left_padding)
        maze_right = int(left_padding + maze_width)
        maze_top = int(top_padding)
        maze_bottom = int(top_padding + maze_height)
        
        # Clear the maze area (all floor)
        for y in range(maze_top, maze_bottom + 1):
            for x in range(maze_left, maze_right + 1):
                self.grid[y, x] = 0
        
        # Create solid maze walls everywhere first
        for y in range(maze_top, maze_bottom + 1):
            for x in range(maze_left, maze_right + 1):
                # Add all possible walls
                self.maze_walls['horizontal'].add((x, y))
                self.maze_walls['horizontal'].add((x, y + 1))
                self.maze_walls['vertical'].add((x, y))
                self.maze_walls['vertical'].add((x + 1, y))
        
        # Now generate maze using algorithm similar to Random labyrinth.py
        visited = set()
        no_wall_tiles = []  # Store connections like in the reference file
        
        # Start from random position
        start_x = random.randint(maze_left + 1, maze_right - 1)
        start_y = random.randint(maze_top + 1, maze_bottom - 1)
        current_pos = (start_x, start_y)
        visited.add(current_pos)
        
        # Generate maze paths
        generation_stack = [current_pos]
        
        while generation_stack:
            current_pos = generation_stack[-1]
            current_x, current_y = current_pos
            
            # Get possible directions (like random_direction() from reference)
            directions = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # down, up, right, left
                new_x, new_y = current_x + dx, current_y + dy
                
                # Check bounds (like is_border() from reference)
                if (maze_left <= new_x <= maze_right and 
                    maze_top <= new_y <= maze_bottom and
                    (new_x, new_y) not in visited):
                    directions.append((new_x, new_y, dx, dy))
            
            if directions:
                # Choose random direction (like random_direction() from reference)
                next_x, next_y, dx, dy = random.choice(directions)
                next_pos = (next_x, next_y)
                
                # Mark as visited
                visited.add(next_pos)
                
                # Create connection in (y,x) order to match pathfinding
                no_wall_tiles.append(((current_y, current_x), (next_y, next_x)))
                
                # Remove wall between current and next position
                if dx > 0:  # Moving right
                    self.maze_walls['vertical'].discard((next_x, current_y))
                elif dx < 0:  # Moving left
                    self.maze_walls['vertical'].discard((current_x, current_y))
                elif dy > 0:  # Moving down
                    self.maze_walls['horizontal'].discard((current_x, next_y))
                elif dy < 0:  # Moving up
                    self.maze_walls['horizontal'].discard((current_x, current_y))
                
                generation_stack.append(next_pos)
            else:
                # Backtrack
                generation_stack.pop()
        
        # Create border walls
        # Top border
        for x in range(maze_left, maze_right + 1):
            self.maze_walls['horizontal'].add((x, maze_top))
            
        # Bottom border
        for x in range(maze_left, maze_right + 1):
            self.maze_walls['horizontal'].add((x, maze_bottom + 1))
            
        # Left border (will have multiple exits)
        for y in range(maze_top, maze_bottom + 1):
            self.maze_walls['vertical'].add((maze_left, y))
            
        # Right border
        for y in range(maze_top, maze_bottom + 1):
            self.maze_walls['vertical'].add((maze_right + 1, y))
        
        # Create ONE entrance on right side (where player enters from middle passage)
        entrance_y = random.randint(maze_top + 2, maze_bottom - 2)
        self.maze_walls['vertical'].discard((maze_right + 1, entrance_y))
        
        # Create ONE exit on left side (where player exits the maze)
        exit_y = random.randint(maze_top + 2, maze_bottom - 2)
        self.maze_walls['vertical'].discard((maze_left, exit_y))
        
        # Store maze coordinates for entrance detection
        self.maze_bounds = {
            'left': maze_left,
            'right': maze_right,
            'top': maze_top,
            'bottom': maze_bottom,
            'entrance_y': entrance_y,
            'exit_y': exit_y
        }
        
        entrances_exits = {'entrance': entrance_y, 'exit': exit_y}
        
        # Add some random additional complexity (like the reference file's random generation)
        for _ in range(10):
            # Randomly remove some internal walls for more open areas
            if no_wall_tiles:
                connection = random.choice(no_wall_tiles)
                pos1, pos2 = connection
                # Our connections are stored in (y,x) order
                y1, x1 = pos1
                y2, x2 = pos2
                
                # Remove adjacent walls for wider passages sometimes
                if random.random() < 0.3:
                    if x1 != x2:  # Horizontal neighbor -> remove vertical wall between them
                        wall_x = max(x1, x2)
                        wall_y = y1  # same row
                        self.maze_walls['vertical'].discard((wall_x, wall_y))
                    else:  # Vertical neighbor -> remove horizontal wall between them
                        wall_x = x1  # same column
                        wall_y = max(y1, y2)
                        self.maze_walls['horizontal'].discard((wall_x, wall_y))
                    # Ensure the connection is present (idempotent)
                    if connection not in no_wall_tiles:
                        no_wall_tiles.append(connection)
        
        # Add barriers to prevent bypassing the maze
        self._add_maze_barriers(maze_left, maze_right, maze_top, maze_bottom, entrance_y, exit_y)
        
        # Add game objects to right chamber  
        self._add_right_chamber_objects()
        
        # Rebuild the full connectivity from the current wall layout to ensure consistency
        self._rebuild_no_wall_edges_from_walls(maze_left, maze_right, maze_top, maze_bottom)
        print(f"🔗 Total {len(self.no_wall_tiles)} connections for pathfinding (rebuilt from walls)")
        
    def _is_wall_between(self, pos1, pos2):
        """Check if there's a maze wall between two adjacent positions."""
        y1, x1 = pos1
        y2, x2 = pos2
        
        # Horizontal movement (left/right)
        if y1 == y2 and abs(x1 - x2) == 1:
            # Check for vertical wall between the positions
            wall_x = max(x1, x2)  # Wall is at the greater x coordinate
            wall_y = y1
            return (wall_x, wall_y) in self.maze_walls['vertical']
        
        # Vertical movement (up/down)
        elif x1 == x2 and abs(y1 - y2) == 1:
            # Check for horizontal wall between the positions
            wall_x = x1
            wall_y = max(y1, y2)  # Wall is at the greater y coordinate
            return (wall_x, wall_y) in self.maze_walls['horizontal']
        
        # Not adjacent positions
        return False

    def _rebuild_no_wall_edges_from_walls(self, maze_left, maze_right, maze_top, maze_bottom):
        """Rebuild self.no_wall_tiles by scanning adjacent cells and keeping only those with no wall between.
        Connections are stored in (y,x) order to match A*.
        """
        edges = []
        for y in range(maze_top, maze_bottom + 1):
            for x in range(maze_left, maze_right + 1):
                if self.grid[y, x] != 0:
                    continue
                # Right neighbor
                nx, ny = x + 1, y
                if nx <= maze_right and self.grid[ny, nx] == 0 and not self._is_wall_between((y, x), (ny, nx)):
                    edges.append(((y, x), (ny, nx)))
                # Down neighbor
                nx, ny = x, y + 1
                if ny <= maze_bottom and self.grid[ny, nx] == 0 and not self._is_wall_between((y, x), (ny, nx)):
                    edges.append(((y, x), (ny, nx)))
        self.no_wall_tiles = edges
        
    def _add_maze_barriers(self, maze_left, maze_right, maze_top, maze_bottom, entrance_y, exit_y):
        """Add barriers around the maze to prevent bypassing."""
        # Add horizontal barriers above and below the maze
        for x in range(1, maze_right + 2):  # Extend slightly beyond maze
            # Top barrier - blocks area above maze
            for y in range(1, maze_top):
                self.grid[y, x] = 1
            
            # Bottom barrier - blocks area below maze  
            for y in range(maze_bottom + 1, GRID_HEIGHT - 1):
                self.grid[y, x] = 1
        
        # Add vertical barriers on left and right sides
        # Left side barrier (but leave the exit open)
        for y in range(1, GRID_HEIGHT - 1):
            if y != exit_y:  # Don't block the exit
                self.grid[y, maze_left - 1] = 1
        
        # Right side barrier (but leave the entrance open)  
        for y in range(1, GRID_HEIGHT - 1):
            if y != entrance_y:  # Don't block the entrance
                self.grid[y, maze_right + 2] = 1
                
    def _add_right_chamber_objects(self):
        """Add prison cell with queen and 3 individual key rooms to right chamber."""
        # Define right chamber boundaries (right_chamber_start = 60)
        right_chamber_start = 60
        chamber_left = right_chamber_start + 2
        chamber_right = GRID_WIDTH - 3
        
        # Prison cell attached to bottom border
        prison_x = (chamber_left + chamber_right) // 2
        prison_y = GRID_HEIGHT - 2  # Against the bottom border
        
        # Create prison cell attached to bottom border (3x2 since bottom is border)
        for dy in range(-1, 1):  # Only go up from border, not down
            for dx in range(-1, 2):
                if dy == -1 or dx == -1 or dx == 1:  # Prison bars (no bottom bars since it's against border)
                    self.grid[prison_y + dy, prison_x + dx] = 3  # 3 = prison bars
        
        # Place queen in center of prison (against bottom border)
        self.grid[prison_y, prison_x] = 4  # 4 = queen
        
        # Create 3 key slots etched into the top wall of right chamber
        self._create_wall_key_slots(chamber_left, chamber_right)
        
        print(f"Added prison cell at ({prison_x}, {prison_y}) and 3 wall key slots")
        
    def _add_master_key_in_left_chamber(self):
        """Add master key at (0,10) and close empty space at (0,9)."""
        # Place master key exactly at (0,10) as shown in image
        master_key_x = 0  # Left wall position
        master_key_y = 10  # Y position 10
        
        # Close the empty space at (0,9) with wall
        if (0 <= 9 < GRID_HEIGHT and 0 <= 0 < GRID_WIDTH):
            self.grid[9, 0] = 1  # Grey wall tile
            
        # Clear the position for master key
        if (0 <= master_key_y < GRID_HEIGHT and 0 <= master_key_x < GRID_WIDTH):
            self.grid[master_key_y, master_key_x] = 0  # Clear floor first
            
        # Place master key at exact position (0,10)
        if (0 <= master_key_y < GRID_HEIGHT and 0 <= master_key_x < GRID_WIDTH):
            self.master_key_pos = (master_key_y, master_key_x)
            self.grid[master_key_y, master_key_x] = 14  # 14 = Master key
            
            print(f"🗝️ Master key placed at exact position ({master_key_x}, {master_key_y})")
            print(f"🧱 Closed empty space at (0, 9) with wall tile")
        
    def _add_guard_at_left_entrance(self, passage_start):
        """Add guard NPC at the entrance to left chamber."""
        # Place guard at the passage entrance to left chamber
        guard_x = passage_start - 1  # Just before the passage starts (at x=24)
        guard_y = GRID_HEIGHT // 2  # Middle of the passage
        
        # Make sure the position is valid
        if (0 <= guard_y < GRID_HEIGHT and 0 <= guard_x < GRID_WIDTH and
            self.grid[guard_y, guard_x] == 0):  # Empty floor
            
            self.guard_pos = (guard_y, guard_x)
            self.grid[guard_y, guard_x] = 12  # 12 = Guard NPC
            
            print(f"Placed guard at passage entrance ({guard_x}, {guard_y})")
        
    def _create_wall_key_slots(self, chamber_left, chamber_right):
        """Create 3 key slots embedded in wall with randomly ordered colored boxes."""
        # Clear any existing slot positions first
        self.slot_positions = []
        
        # Position key slots in the top wall (y=0 is the border wall)
        slot_y = 0  # In the top border wall itself
        
        # Calculate positions for 3 slots spread across the top wall
        slot_spacing = (chamber_right - chamber_left - 2) // 4  # Distribute evenly
        
        # Define colored boxes: 6=Red box, 7=Green box, 8=Blue box
        colored_boxes = [6, 7, 8]  # R, G, B
        
        # Better randomization - use time-based seed with more entropy
        import time
        import os
        random_seed = int(time.time() * 1000000) + os.getpid()  # More entropy
        random.seed(random_seed)
        
        # Shuffle until we don't get R-G-B order (6, 7, 8)
        max_attempts = 20
        for _ in range(max_attempts):
            random.shuffle(colored_boxes)
            if colored_boxes != [6, 7, 8]:  # Avoid R-G-B order
                break
        
        # If we somehow still got R-G-B, force a different order
        if colored_boxes == [6, 7, 8]:
            colored_boxes = [8, 6, 7]  # Force B-R-G instead
        
        for i in range(3):
            slot_x = chamber_left + 2 + (i * slot_spacing)
            self.slot_positions.append((slot_y, slot_x))
            
            # Create key slot embedded in the wall (replace wall with slot)
            self.grid[slot_y, slot_x] = 5  # Key slot (gold background)
            
            # Place colored box INSIDE the slot (same position)
            box_y, box_x = slot_y, slot_x  # Same position as the slot
            
            # Track this box as moveable (the visual will show box on top of slot)
            self.box_positions[(box_y, box_x)] = colored_boxes[i]
        
        # Store the original box positions for queen challenge
        self.original_box_positions = self.box_positions.copy()
        
        # Store the box arrangement for reference
        box_colors = ['Red', 'Green', 'Blue']
        color_order = [box_colors[box - 6] for box in colored_boxes]
        
        print(f"Created 3 key slots embedded in wall at x positions: {[pos[1] for pos in self.slot_positions]}")
        print(f"Colored boxes randomly arranged as: {color_order}")
        print(f"Box positions tracked: {self.box_positions}")
        print(f"Secret keys for this session: {self.secret_keys}")
        print(f"🏰 QUEEN CHALLENGE: Original box positions stored for final challenge")
        
    def _generate_secret_keys(self):
        """Generate 3 random 5-digit keys for this game session."""
        import time
        import os
        random.seed(int(time.time() * 1000000) + os.getpid())
        
        keys = []
        for i in range(3):
            # Generate 5-digit number (10000-99999)
            key = random.randint(10000, 99999)
            keys.append(key)
        
        return keys
        
    def _check_rgb_order(self):
        """Check if boxes are arranged in R-G-B order in the slots."""
        if self.keys_released:
            return  # Already released keys
            
        # Check if each slot has the correct color box
        slot_boxes = []
        for slot_pos in self.slot_positions:
            if slot_pos in self.box_positions:
                slot_boxes.append(self.box_positions[slot_pos])
            else:
                slot_boxes.append(None)  # Empty slot
        
        # Debug: Print current arrangement 
        print(f"Current slot arrangement: {slot_boxes}")
        print(f"Looking for R-G-B order: [6, 7, 8]")
        print(f"Slot positions: {self.slot_positions}")
        print(f"Box positions: {self.box_positions}")
        
        # IMPORTANT: Boxes must be placed EXACTLY in slot positions [(0,64), (0,67), (0,70)]
        # Current arrangement needs to be [6, 7, 8] (Red, Green, Blue)
        if slot_boxes == [6, 7, 8]:
            print("🎉 WIN CONDITION MET! R-G-B order achieved!")
        else:
            print(f"❌ Win condition not met. Need Red(6) at (0,64), Green(7) at (0,67), Blue(8) at (0,70)")
            if len(slot_boxes) >= 3:
                print(f"   Currently: slot (0,64)={slot_boxes[0] if slot_boxes[0] else 'Empty'}," + 
                      f" slot (0,67)={slot_boxes[1] if slot_boxes[1] else 'Empty'}," +
                      f" slot (0,70)={slot_boxes[2] if slot_boxes[2] else 'Empty'}")
            else:
                print(f"   ERROR: Only {len(slot_boxes)} slots found, should be 3!")
                
            # Show where Red box is currently
            for pos, box_type in self.box_positions.items():
                if box_type == 6:  # Red box
                    print(f"   Red box is currently at {pos}, needs to move to (0, 64)")
        
        # Check if arrangement is Red-Green-Blue (6-7-8)
        if slot_boxes == [6, 7, 8]:
            self._release_keys()
            
    def _release_keys(self):
        """Release the secret keys when R-G-B order is achieved."""
        if self.keys_released:
            return
            
        self.keys_released = True
        print("🎉 SUCCESS! Boxes arranged in R-G-B order!")
        print("🗝️ SECRET KEYS RELEASED:")
        for i, key in enumerate(self.secret_keys):
            color_name = ['Red', 'Green', 'Blue'][i]
            print(f"   {color_name} Key: {key}")
        
        # Spawn visual key objects in the game world
        self._spawn_key_objects()
        
    def _spawn_key_objects(self):
        """Spawn visual key objects in front of RGB slots as small circles."""
        # Spawn keys in front of each slot (one position below the wall slots)
        key_types = [9, 10, 11]  # Red key, Green key, Blue key
        
        for i, slot_pos in enumerate(self.slot_positions):
            slot_y, slot_x = slot_pos
            # Position keys in front of slots (one tile below)
            key_y, key_x = slot_y + 1, slot_x
            
            # Make sure position is valid and empty
            if (0 <= key_y < GRID_HEIGHT and 0 <= key_x < GRID_WIDTH and
                self.grid[key_y, key_x] == 0):  # Empty floor
                
                # Store key object position and value (don't place in grid for circle rendering)
                self.key_objects[(key_y, key_x)] = {
                    'type': key_types[i],
                    'value': self.secret_keys[i]
                }
                
                color_name = ['Red', 'Green', 'Blue'][i]
                print(f"Spawned {color_name} key ({self.secret_keys[i]}) in front of slot at ({key_x}, {key_y})")
                
    def _try_collect_key(self, position):
        """Try to collect a key at the given position."""
        # Convert list to tuple for dictionary lookup
        if isinstance(position, list):
            position = tuple(position)
            
        if position in self.key_objects:
            key_data = self.key_objects[position]
            key_type = key_data['type']
            key_value = key_data['value']
            
            # Add to player's inventory
            self.collected_keys.append({
                'type': key_type,
                'value': key_value,
                'color': ['Red', 'Green', 'Blue'][key_type - 9]
            })
            
            # Remove from world
            del self.key_objects[position]
            
            color_name = ['Red', 'Green', 'Blue'][key_type - 9]
            print(f"✅ Collected {color_name} key: {key_value}")
                
            # Start timer to display key values
            self.key_display_timer = pygame.time.get_ticks()
                
            # Show info popup for key collection
            if len(self.collected_keys) == 3:
                print("🎉 All keys collected! Ready for guard challenge!")
                
                self._show_info_popup(
                    "🎉 ALL KEYS COLLECTED!",
                    [
                        "You have collected all 3 RGB keys!",
                        f"Red Key: {self.collected_keys[0]['value']}",
                        f"Green Key: {self.collected_keys[1]['value']}",
                        f"Blue Key: {self.collected_keys[2]['value']}",
                        "",
                        "🛡️ Next: Find the guard in the middle passage",
                        "The guard will test your memory of these keys!"
                    ],
                    auto_close_seconds=3
                )
            else:
                self._show_info_popup(
                    f"🔑 {color_name.upper()} KEY COLLECTED!",
                    [
                        f"Key Value: {key_value}",
                        f"Progress: {len(self.collected_keys)}/3 keys",
                        "",
                        "Keep exploring to find the remaining keys!"
                    ],
                    auto_close_seconds=2
                )
                
    def _collect_navigation_map(self, position):
        """Check if player is collecting the navigation map."""
        # Convert list to tuple if needed
        if isinstance(position, list):
            position = tuple(position)
        y, x = position
        
        if (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH and
            self.grid[y, x] == 13):  # Navigation map
            
            # Collect the map
            self.grid[y, x] = 0  # Remove from grid
            self.maze_map_collected = True  # Mark as collected
            
            print("🗺️ Collected MAZE NAVIGATION MAP!")
            print("📋 Press M to read the navigation instructions")
            print(f"🧭 Map contains {len(self.inverted_instructions) + 1} steps to reach the queen")
            print("⚠️  WARNING: Instructions may be... unconventional.")
            
            self._show_info_popup(
                "🗺️ MAZE NAVIGATION MAP FOUND!",
                [
                    "You discovered a mysterious navigation map!",
                    "",
                    f"📋 Press 'M' to read the navigation instructions",
                    f"🧭 Contains {len(self.inverted_instructions) + 1} steps to reach the queen",
                    "",
                    "⚠️ WARNING: Instructions may be unconventional...",
                    "🔍 The maze walls will become invisible when you enter",
                    "📍 Follow the map directions carefully!",
                    "",
                    "This map will guide you through the maze safely."
                ],
                auto_close_seconds=5
            )
            
            # Show the inverted navigation instructions now that map is collected
            print("\n🧭 MAZE NAVIGATION INSTRUCTIONS:")
            if hasattr(self, 'maze_solution_path') and self.maze_solution_path:
                path = self.maze_solution_path
                direction_map = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}
                invert_map = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
                
                for i in range(len(path) - 1):
                    current = path[i]
                    next_pos = path[i + 1]
                    dy = next_pos[0] - current[0]
                    dx = next_pos[1] - current[1]
                    actual_direction = direction_map.get((dy, dx))
                    if actual_direction:
                        inverted_direction = invert_map[actual_direction]
                        print(f"🧭 Step {i+1}: Move from {current} to {next_pos} = {actual_direction} → inverted to {inverted_direction}")
            print("📍 Follow these inverted directions to navigate the maze!")
                
    def _check_master_key_interaction(self, position):
        """Check if player is trying to interact with the master key."""
        if not self.master_key_pos or self.master_key_collected:
            return False
            
        # Check if player is adjacent to or on master key
        key_y, key_x = self.master_key_pos
        player_y, player_x = position
        
        distance = abs(player_y - key_y) + abs(player_x - key_x)
        
        if distance <= 1:  # Adjacent to the key
            # Check if all RGB keys are collected (prerequisite)
            if len(self.collected_keys) < 3:
                print("❌ MASTER KEY: Need all 3 RGB keys first!")
                print("🔍 Collect Red, Green, and Blue keys before attempting master key challenge")
                
                self._show_info_popup(
                    "🗝️ MASTER KEY BLOCKED!",
                    [
                        "You need all 3 RGB keys first!",
                        f"Current Progress: {len(self.collected_keys)}/3 keys",
                        "",
                        "🔍 Find and collect:",
                        "• Red Key",
                        "• Green Key", 
                        "• Blue Key",
                        "",
                        "Then return to attempt the Master Key challenge!"
                    ],
                    auto_close_seconds=4
                )
                return False
                
            # Initiate master key challenge
            self._initiate_master_key_challenge()
            return True
            
        return False
        
    def _initiate_master_key_challenge(self):
        """Start the master key mathematical challenge."""
        if self.master_key_challenge_active:
            return
            
        # Use the same RGB key digits but apply mathematical operations
        if len(self.collected_keys) < 3:
            return
            
        # Select random digits from each key (like guard challenge)
        import random
        rgb_digits = []
        digit_info = []
        
        for key_data in self.collected_keys:
            key_value = str(key_data['value'])  # Convert to string for digit access
            color = key_data['color']
            digit_pos = random.randint(0, 4)  # Random position (0-4 for 5-digit number)
            selected_digit = int(key_value[digit_pos])
            
            rgb_digits.append(selected_digit)
            digit_info.append({
                'color': color,
                'digit': selected_digit,
                'position': digit_pos + 1,  # 1-indexed for display
                'key_value': key_value
            })
        
        # Choose random operation with variety
        operations = [
            'add_simple',        # Just add the 3 digits
            'multiply_simple',   # Just multiply the 3 digits
            'add_divide',        # Add then divide by random number
            'add_subtract',      # Add then subtract random number
            'multiply_divide',   # Multiply then divide by random number
            'add_multiply'       # Add then multiply by random number
        ]
        self.master_key_operation = random.choice(operations)
        
        # Generate random parameters for certain operations (ensuring 3+ digit results)
        self.random_divisor = random.randint(2, 3)  # Smaller divisor for larger results
        self.random_subtractor = random.randint(-50, -10)  # Negative subtractor = addition
        self.random_multiplier = random.randint(15, 25)  # Larger multiplier
        
        # Calculate the expected password based on operation (no padding required)
        result = self._calculate_master_key_result(rgb_digits, self.master_key_operation)
        self.master_key_expected_password = str(result)  # No zero padding
        
        # Create human-readable operation description (with actual values for console)
        operation_str_add = " + ".join([str(d) for d in rgb_digits])
        operation_str_mul = " × ".join([str(d) for d in rgb_digits])
        
        if self.master_key_operation == 'add_simple':
            self.master_key_operation_text = f"Add all digits: {operation_str_add}"
            self.master_key_instruction_text = f"Add digit #{digit_info[0]['position']} from {digit_info[0]['color']} key, digit #{digit_info[1]['position']} from {digit_info[1]['color']} key, digit #{digit_info[2]['position']} from {digit_info[2]['color']} key"
        elif self.master_key_operation == 'multiply_simple':
            self.master_key_operation_text = f"Multiply all digits: {operation_str_mul}"
            self.master_key_instruction_text = f"Multiply digit #{digit_info[0]['position']} from {digit_info[0]['color']} key × digit #{digit_info[1]['position']} from {digit_info[1]['color']} key × digit #{digit_info[2]['position']} from {digit_info[2]['color']} key"
        elif self.master_key_operation == 'add_divide':
            self.master_key_operation_text = f"Add digits then ÷{self.random_divisor}: ({operation_str_add}) ÷ {self.random_divisor}"
            self.master_key_instruction_text = f"Add all 3 digits then divide by {self.random_divisor}"
        elif self.master_key_operation == 'add_subtract':
            # Since subtractor is negative, display it as addition
            actual_addend = abs(self.random_subtractor)
            self.master_key_operation_text = f"Add digits then +{actual_addend}: ({operation_str_add}) + {actual_addend}"
            self.master_key_instruction_text = f"Add all 3 digits then add {actual_addend}"
        elif self.master_key_operation == 'multiply_divide':
            self.master_key_operation_text = f"Multiply digits then ÷{self.random_divisor}: ({operation_str_mul}) ÷ {self.random_divisor}"
            self.master_key_instruction_text = f"Multiply all 3 digits then divide by {self.random_divisor}"
        elif self.master_key_operation == 'add_multiply':
            self.master_key_operation_text = f"Add digits then ×{self.random_multiplier}: ({operation_str_add}) × {self.random_multiplier}"
            self.master_key_instruction_text = f"Add all 3 digits then multiply by {self.random_multiplier}"
        
        print(f"")
        # Display selected digits and the mathematical operation
        print(f"🔢 Selected Digits from Your Keys:")
        for info in digit_info:
            print(f"   {info['color']} Key ({info['key_value']}): Digit #{info['position']} = {info['digit']}")
        
        print(f"📊 Operation: {self.master_key_operation_text}")
        print(f"🎯 Enter the result as your password (1-5 digits)")
        
        # Store digit info for UI popup
        self.master_key_digit_info = digit_info
        
        # No padding hint needed anymore - just enter the result directly
        
        self.master_key_challenge_active = True
        self.master_key_password_ui_active = True
        self.master_key_password_input = ""
        
    def _calculate_master_key_result(self, digits, operation):
        """Calculate the result exactly as displayed to the user."""
        if operation == 'add_simple':
            # Add the 3 digits exactly as shown
            return sum(digits)
            
        elif operation == 'multiply_simple':
            # Multiply the 3 digits exactly as shown
            result = 1
            for digit in digits:
                result *= digit
            return result
            
        elif operation == 'add_divide':
            # Add the digits then divide exactly as shown
            total = sum(digits)
            result = total // self.random_divisor
            return max(result, 1)  # Ensure at least 1
            
        elif operation == 'add_subtract':
            # Add the digits then subtract/add exactly as shown
            total = sum(digits)
            result = total - self.random_subtractor  # With negative subtractor, this becomes addition
            return result
            
        elif operation == 'multiply_divide':
            # Multiply the digits then divide exactly as shown
            product = 1
            for digit in digits:
                product *= digit
            result = product // self.random_divisor
            return max(result, 1)  # Ensure at least 1
            
        elif operation == 'add_multiply':
            # Add the digits then multiply exactly as shown
            total = sum(digits)
            result = total * self.random_multiplier
            return result
            
        else:
            return 1111  # Fallback
                
    def _check_queen_interaction(self, position):
        """Check if player is trying to interact with the queen for final challenge."""
        if self.queen_released:
            return False
            
        # Find queen position
        queen_pos = None
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y, x] == 4:  # 4 = queen
                    queen_pos = (y, x)
                    break
            if queen_pos:
                break
                
        if not queen_pos:
            return False
            
        # Check if player is adjacent to queen
        queen_y, queen_x = queen_pos
        player_y, player_x = position
        
        distance = abs(player_y - queen_y) + abs(player_x - queen_x)
        
        if distance <= 1:  # Adjacent to the queen
            # Check prerequisites: must have master key
            if not self.master_key_collected:
                print("🔒 PRISONER: 'You must prove your mastery first. Find the master key.'")
                
                self._show_info_popup(
                    "🔒 PRISONER SPEAKS!",
                    [
                        "The imprisoned figure calls out to you:",
                        "",
                        "'You must prove your mastery first!'",
                        "'Find the master key to show your worth!'",
                        "",
                        "🗝️ Find the Master Key first",
                        "Then return to attempt the final challenge"
                    ],
                    auto_close_seconds=4
                )
                return False
                
            # If challenge already started, check box positions
            if self.queen_challenge_active:
                if not self._check_boxes_in_original_positions():
                    print("🔒 PRISONER: 'The colored boxes are not in their original positions yet.'")
                    print("📦 REMINDER: Move the RGB boxes back where you first saw them!")
                    current_positions = list(self.box_positions.keys())
                    original_positions = list(self.original_box_positions.keys())
                    print(f"📍 Current positions: {current_positions}")
                    print(f"🎯 Target positions: {original_positions}")
                    
                    self._show_info_popup(
                        "🔒 PRISONER'S MEMORY TEST!",
                        [
                            "The imprisoned figure shakes their head:",
                            "",
                            "'The colored boxes are not in their original positions!'",
                            "",
                            "📦 PHASE 1: Box Rearrangement Required",
                            "• Move RGB boxes back to where you first saw them",
                            "• Use 'F' key to grab/release boxes",
                            "• Remember the ORIGINAL layout from game start",
                            "",
                            f"📍 Current positions: {current_positions}",
                            f"🎯 Target positions: {original_positions}",
                            "",
                            "Only then can you proceed to the final password!"
                        ],
                        auto_close_seconds=6
                    )
                    return False
                else:
                    # Boxes are correctly positioned, allow password entry
                    print("✅ PRISONER: 'Perfect! The boxes are back in their original positions.'")
                    print("🔒 PRISONER: 'Now you may enter the final password.'")
                    self.queen_password_ui_active = True
                    return True
            else:
                # Initiate queen challenge
                self._initiate_queen_challenge()
                return True
            
        return False
        
    def _initiate_queen_challenge(self):
        """Start the queen release challenge."""
        if self.queen_challenge_active:
            return
            
        print(f"")
        print(f"🔒 PRISONER: 'Brave adventurer, you have come far...'")
        print(f"🔒 PRISONER: 'To free me, you must prove your memory and skill.'")
        print(f"🔒 PRISONER: 'First, return the colored boxes to their ORIGINAL positions.'")
        print(f"🔒 PRISONER: 'Only then may you attempt the final password.'")
        print(f"")
        print(f"🏰 PRISONER CHALLENGE ACTIVATED!")
        print(f"📦 PHASE 1: Rearrange RGB boxes to their original positions")
        print(f"🔐 PHASE 2: Enter the final password (after Phase 1 is complete)")
        print(f"⚠️  WARNING: You have only 3 password attempts!")
        
        self._show_info_popup(
            "🔒 PRISONER'S FINAL CHALLENGE!",
            [
                "The imprisoned figure speaks urgently:",
                "",
                "'Brave adventurer, you have come far...'",
                "'To free me, you must prove your memory and skill!'",
                "",
                "📦 PHASE 1: Return boxes to ORIGINAL positions",
                "🔐 PHASE 2: Enter the final password",
                "",
                "⚠️ WARNING: Only 3 password attempts!",
                "Remember your journey and prove your worth!"
            ],
            auto_close_seconds=7
        )
        
        self.queen_challenge_active = True
        
        # Generate the expected password: random digit from master key + reversed guard password
        self._generate_queen_password()
        
    def _generate_queen_password(self):
        """Generate the queen password from specific master key digit position + reversed guard password."""
        if not hasattr(self, 'master_key_expected_password') or not hasattr(self, 'guard_expected_passwords'):
            print("❌ ERROR: Cannot generate queen password - missing prerequisites")
            return
            
        # Get ANY random digit from master key password (like guard riddle)
        import random
        master_key_str = str(self.master_key_expected_password)
        
        # Choose from ANY available position (1 to length of string)
        available_positions = list(range(1, len(master_key_str) + 1))  # 1-indexed positions
        
        if not available_positions:
            # Fallback if master key is empty (shouldn't happen)
            available_positions = [1]
            master_key_str = "123"  # Emergency fallback
            
        # Pick random position and get that digit
        chosen_position = random.choice(available_positions)
        random_master_digit = master_key_str[chosen_position - 1]  # Convert to 0-indexed
        
        # Store the chosen position for UI display
        self.queen_chosen_digit_position = chosen_position
        self.queen_chosen_digit_value = random_master_digit
        
        # Get reversed guard password
        guard_password = self.guard_expected_passwords[0] if self.guard_expected_passwords else "00000"
        reversed_guard_password = guard_password[::-1]
        
        # Combine: specific master key digit + reversed guard password
        self.queen_expected_password = random_master_digit + reversed_guard_password
        
        print(f"🏰 QUEEN PASSWORD GENERATED!")
        print(f"🔢 Components: {self._ordinal(chosen_position)} digit from master key + Reversed guard password")
        print(f"💡 HINT: Master key was '{self.master_key_expected_password}', Guard password was '{guard_password}'")
        print(f"📍 Using {self._ordinal(chosen_position)} digit '{random_master_digit}' from master key")
        
    def _check_boxes_in_original_positions(self):
        """Check if all RGB boxes are back in their original positions."""
        if not self.original_box_positions:
            return False
            
        # Compare current positions with original positions
        return self.box_positions == self.original_box_positions
        
    def _handle_queen_password_input(self, event_key):
        """Handle password input during queen challenge."""
        if not self.queen_password_ui_active:
            return False
            
        # Handle number keys 0-9
        if pygame.K_0 <= event_key <= pygame.K_9:
            digit = str(event_key - pygame.K_0)
            self.queen_password_input += digit
            print(f"👑 QUEEN: Entered digit {digit} (Password: {'*' * len(self.queen_password_input)})")
            
        elif event_key == pygame.K_BACKSPACE:
            if self.queen_password_input:
                self.queen_password_input = self.queen_password_input[:-1]
                print(f"👑 QUEEN: Deleted digit (Password: {'*' * len(self.queen_password_input)})")
            
        elif event_key == pygame.K_RETURN or event_key == pygame.K_KP_ENTER:
            if len(self.queen_password_input) >= 1:
                self._process_queen_password(self.queen_password_input)
            else:
                print(f"👑 QUEEN: Please enter at least 1 digit!")
                
        elif event_key == pygame.K_ESCAPE:
            # Cancel password entry
            self.queen_password_ui_active = False
            self.queen_password_input = ""
            print("👑 QUEEN: 'Password entry cancelled. Approach me again when ready.'")
            
    def _process_queen_password(self, password):
        """Process the entered queen password."""
        self.queen_attempts += 1
        
        if password == self.queen_expected_password:
            print(f"✅ QUEEN: Correct! The final password is accepted!")
            print(f"🎉 Password '{password}' matches the expected combination!")
            
            # Queen is released!
            self.queen_released = True
            self.queen_password_ui_active = False
            self.queen_password_input = ""
            
            print(f"")
            print(f"👑 QUEEN: 'Thank you, brave hero! You have freed me from this prison!'")
            print(f"👑 QUEEN: 'Your memory, skill, and determination are legendary!'")
            print(f"")
            
            self._show_info_popup(
                "🏆 VICTORY! QUEEN FREED!",
                [
                    "👑 QUEEN: 'Thank you, brave hero!'",
                    "👑 QUEEN: 'You have freed me from this prison!'",
                    "👑 QUEEN: 'Your memory, skill, and determination are legendary!'",
                    "",
                    "🏆 GAME COMPLETED! All challenges conquered:",
                    "✅ RGB Keys collected",
                    "✅ Guard challenge passed", 
                    "✅ Master Key obtained",
                    "✅ Boxes rearranged to original positions",
                    "✅ Final password solved",
                    "",
                    "🎉 You are the ultimate dungeon master!"
                ],
                auto_close_seconds=0  # Manual close for victory
            )
            
            # Remove queen from grid (she's free!)
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.grid[y, x] == 4:  # Find queen
                        self.grid[y, x] = 0  # Replace with floor
                        break
            
        else:
            print(f"❌ QUEEN: Incorrect! Expected '{self.queen_expected_password}', got '{password}'")
            print(f"👑 QUEEN: 'That is not the correct combination. Think carefully...'")
            print(f"🔢 Remember: Random digit from master key + Reversed guard password")
            print(f"💡 You have {3 - self.queen_attempts} attempts remaining!")
            
            if self.queen_attempts >= 3:
                print(f"")
                print(f"💀 GAME OVER: Maximum password attempts exceeded!")
                print(f"👑 QUEEN: 'I'm sorry, but you have failed the final test...'")
                print(f"🔒 The queen remains imprisoned forever.")
                print(f"")
                print(f"Exiting game...")
                pygame.quit()
                exit()
            
            # Reset for retry
            self.queen_password_input = ""
        
    def _check_guard_challenge(self, new_pos):
        """Check if player is trying to pass the guard and initiate challenge."""
        if self.guard_passed or not self.guard_pos:
            return False  # Already passed or no guard
            
        guard_y, guard_x = self.guard_pos
        new_y, new_x = new_pos
        
        # Block ANY movement into the left chamber area (x <= 22)
        # Left chamber ends at x=20, so block anything x <= 22 to be safe
        if new_x <= 22:
            print(f"DEBUG: Blocking movement to x={new_x} (left chamber boundary)")
            return self._initiate_guard_challenge()
            
        return False
        
    def _initiate_guard_challenge(self):
        """Start the guard challenge to verify keys."""
        if len(self.collected_keys) < 3:
            print("🛡️ GUARD: 'Halt! You need 3 keys to pass. Come back when you have them all.'")
            
            self._show_info_popup(
                "🛡️ GUARD BLOCKS THE WAY!",
                [
                    "The guard stops you from proceeding!",
                    f"You need 3 RGB keys to pass, but you only have {len(self.collected_keys)}",
                    "",
                    "🔍 Find the missing keys:",
                    "• Look for Red, Green, and Blue key slots",
                    "• Keys are in the right chamber slots",
                    "• Move the colored boxes to reveal keys",
                    "",
                    "Return when you have all 3 keys!"
                ],
                auto_close_seconds=5
            )
            return True  # Block movement
            
        print("🛡️ GUARD: 'Stop! I need to verify your keys before you can enter.'")
        print("🛡️ GUARD: 'I will ask for specific digits from each key.'")
        
        # Generate 5 random digit challenges across all keys
        import random
        self.guard_digit_challenges = []
        password_digits = []
        
        # Generate 5 digit requests randomly distributed across the 3 keys
        for challenge_num in range(5):
            # Pick a random key
            key_index = random.randint(0, len(self.collected_keys) - 1)
            key_data = self.collected_keys[key_index]
            key_value = str(key_data['value'])
            color = key_data['color']
            
            # Ask for a random digit position (1-5 for 5-digit number)
            digit_pos = random.randint(0, 4)  # 0-based index
            digit_value = key_value[digit_pos]
            
            self.guard_digit_challenges.append({
                'challenge_num': challenge_num + 1,
                'color': color,
                'key_value': key_value,
                'digit_position': digit_pos + 1,  # 1-based for display
                'digit_value': digit_value
            })
            
            password_digits.append(digit_value)
        
        # The correct password is all requested digits combined
        self.guard_expected_passwords = [''.join(password_digits)]
        
        # Print only questions to console (no hints)
        print("🛡️ GUARD: 'Answer these questions in order:'")
        for challenge in self.guard_digit_challenges:
            print(f"   {challenge['challenge_num']}. What is the {self._ordinal(challenge['digit_position'])} digit of your {challenge['color']} key?")
        print("🛡️ GUARD: 'Enter all 5 answers combined as your password.'")
        
        # Show password UI
        self.guard_password_ui_active = True
        self.guard_password_input = ""
        
        return True  # Block movement until password entered
        
    def _ordinal(self, n):
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
        
    def _run_key_verification(self):
        """Run the key verification challenge."""
        import random
        
        print("\n=== KEY VERIFICATION CHALLENGE ===")
        print("🛡️ GUARD: 'I need to verify each of your keys.'")
        print("🛡️ GUARD: 'I will ask for specific digits from each key.'")
        print("🛡️ GUARD: 'Press number keys (1-5) to answer, ENTER to confirm.'")
        
        # Generate questions for each key
        self.guard_challenge_questions = []
        for i, key_data in enumerate(self.collected_keys):
            key_value = str(key_data['value'])  # Convert to string for digit access
            color = key_data['color']
            
            # Ask for a random digit position (0-4 for 5-digit number)
            digit_pos = random.randint(0, 4)
            expected_digit = key_value[digit_pos]
            
            self.guard_challenge_questions.append({
                'color': color,
                'key_value': key_value,
                'digit_pos': digit_pos,
                'expected_digit': expected_digit
            })
        
        # Start challenge
        self.guard_challenge_active = True
        self.guard_current_question = 0
        self._show_current_guard_question()
        return True  # Block movement until challenge complete
        
    def _show_current_guard_question(self):
        """Show the current guard challenge question."""
        if self.guard_current_question >= len(self.guard_challenge_questions):
            return
            
        q = self.guard_challenge_questions[self.guard_current_question]
        print(f"\n🛡️ GUARD: 'Question {self.guard_current_question + 1}/3:'")
        print(f"🛡️ GUARD: 'Your {q['color']} key - what is digit #{q['digit_pos'] + 1}?'")
        print(f"         (Hint: Your {q['color']} key is {q['key_value']})")
        print("         Press the digit (1-5 keys) then ENTER:")
        
    def _handle_guard_password_input(self, event_key):
        """Handle password input during guard challenge."""
        if not self.guard_password_ui_active:
            return False
            
        # Handle number keys 0-9
        if pygame.K_0 <= event_key <= pygame.K_9:
            digit = str(event_key - pygame.K_0)
            self.guard_password_input += digit
            print(f"Password input: {self.guard_password_input}")
            
        elif event_key == pygame.K_BACKSPACE:
            # Remove last character
            if self.guard_password_input:
                self.guard_password_input = self.guard_password_input[:-1]
                print(f"Password input: {self.guard_password_input}")
                
        elif event_key == pygame.K_RETURN:
            # Check password
            self._verify_guard_password()
            
        elif event_key == pygame.K_ESCAPE:
            # Cancel password entry
            self.guard_password_ui_active = False
            self.guard_password_input = ""
            print(f"🛡️ GUARD: 'Come back when you're ready.'")
                
    def _handle_master_key_password_input(self, event_key):
        """Handle password input during master key challenge."""
        if not self.master_key_password_ui_active:
            return False
            
        # Handle number keys 0-9
        if pygame.K_0 <= event_key <= pygame.K_9:
            if len(self.master_key_password_input) < 5:  # Max 5 digits
                digit = str(event_key - pygame.K_0)
                self.master_key_password_input += digit
                print(f"🗝️ MASTER KEY: Entered digit {digit} (Password: {'*' * len(self.master_key_password_input)})")
                
        elif event_key == pygame.K_BACKSPACE:
            if self.master_key_password_input:
                self.master_key_password_input = self.master_key_password_input[:-1]
                print(f"🗝️ MASTER KEY: Deleted digit (Password: {'*' * len(self.master_key_password_input)})")
            
        elif event_key == pygame.K_RETURN or event_key == pygame.K_KP_ENTER:
            if len(self.master_key_password_input) >= 1:  # Accept 1-5 digits
                self._process_master_key_password(self.master_key_password_input)
            else:
                print(f"🗝️ MASTER KEY: Please enter at least 1 digit!")
                
        elif event_key == pygame.K_ESCAPE:
            # Cancel password entry
            self.master_key_password_ui_active = False
            self.master_key_password_input = ""
            self.master_key_challenge_active = False
            print("🗝️ MASTER KEY: 'Challenge cancelled. Try again when ready.'")
                
    def _process_master_key_password(self, password):
        """Process the entered master key password."""
        if password == self.master_key_expected_password:
            print(f"✅ MASTER KEY: Correct! Mathematical operation solved!")
            print(f"🎉 {self.master_key_operation_text} = {password}")
            
            # Collect the master key
            self.master_key_collected = True
            self.master_key_password_ui_active = False
            self.master_key_challenge_active = False
            self.master_key_password_input = ""
            
            # Remove key from grid
            if self.master_key_pos:
                key_y, key_x = self.master_key_pos
                self.grid[key_y, key_x] = 0
            
            print(f"🗝️ MASTER KEY COLLECTED! You are now the ultimate key master!")
            print(f"👑 This key unlocks the deepest secrets of the dungeon...")
            
            self._show_info_popup(
                "🗝️ MASTER KEY OBTAINED!",
                [
                    "🎉 Mathematical challenge solved perfectly!",
                    f"✅ Correct answer: {self.master_key_expected_password}",
                    "",
                    "🗝️ MASTER KEY COLLECTED!",
                    "You are now the ultimate key master!",
                    "",
                    "👑 This key unlocks the deepest secrets...",
                    "🏰 You can now approach the imprisoned queen!",
                    "",
                    "Navigate through the maze and free the prisoner!"
                ],
                auto_close_seconds=6
            )
            
        else:
            print(f"❌ MASTER KEY: Incorrect! Expected {self.master_key_expected_password}, got {password}")
            print(f"🔢 Remember: {self.master_key_operation_text}")
            print(f"💡 Try again with the correct calculation!")
            
            # Reset for retry
            self.master_key_password_input = ""
        
    def _verify_guard_password(self):
        """Verify the entered password against collected key values."""
        if len(self.guard_password_input) != 5:
            print(f"❌ Password must be exactly 5 digits! You entered {len(self.guard_password_input)} digits.")
            print("🛡️ GUARD: 'I need exactly 5 digits as specified!'")
            self.guard_password_input = ""
            return
            
        if self.guard_password_input in self.guard_expected_passwords:
            # Password correct!
            print(f"✅ Password '{self.guard_password_input}' accepted!")
            print("🛡️ GUARD: 'All 5 digits correct! You may pass.'")
            print("🎉 GUARD PASSED! Access to left chamber granted!")
            
            self._show_info_popup(
                "🛡️ GUARD CHALLENGE PASSED!",
                [
                    f"✅ Password '{self.guard_password_input}' accepted!",
                    "",
                    "🛡️ GUARD: 'All 5 digits correct! You may pass.'",
                    "",
                    "🎉 GUARD PASSED! Access to left chamber granted!",
                    "🗝️ Now search for the Master Key",
                    "🏰 Navigate through the maze to reach the queen",
                    "",
                    "Your memory skills have been proven!",
                    "Continue your quest to free the prisoner."
                ],
                auto_close_seconds=5
            )
            
            # Grant access
            self.guard_passed = True
            self.guard_password_ui_active = False
            self.guard_password_input = ""
            
            # Generate maze solution and spawn map (but don't start step tracking yet)
            self._find_maze_solution_path()
            self._spawn_maze_navigation_map()
            
            print(f"🔍 Navigate to the maze entrance to start step tracking!")
            print(f"📍 Look for the grey pillars area at the maze entrance")
            
        else:
            # Password wrong
            print(f"❌ Password '{self.guard_password_input}' rejected!")
            print("🛡️ GUARD: 'Wrong digits! Check the terminal for the correct requirements.'")
            print(f"Expected: {self.guard_expected_passwords[0]} (but don't tell the player this)")
            
            # Clear input but keep UI active
            self.guard_password_input = ""
        
    def _find_maze_solution_path(self):
        """Find the solution path through the maze using A* pathfinding."""
        # Use ACTUAL maze entrance coordinates from generation
        start_pos = None
        if hasattr(self, 'maze_bounds'):
            # Start from one step BEFORE the entrance to match manual navigation
            entrance_y = self.maze_bounds['entrance_y']
            entrance_x = self.maze_bounds['right']
            # Manual navigation starts from (entrance_y, entrance_x + 1) and steps LEFT to entrance
            start_pos = (entrance_y, entrance_x + 1)
            print(f"🚪 Found ACTUAL maze entrance at ({entrance_y}, {entrance_x})")
            print(f"🎯 Starting from one step before: {start_pos} → entrance")
        
        if start_pos is None:
            print(f"❌ Could not determine maze entrance coordinates!")
            return
            
        # Use ACTUAL maze exit coordinates as goal (not the queen!)
        goal_pos = None
        if hasattr(self, 'maze_bounds'):
            # The goal should be the maze EXIT, not the queen in right chamber
            exit_y = self.maze_bounds['exit_y']
            exit_x = self.maze_bounds['left']
            goal_pos = (exit_y, exit_x)
            print(f"🚪 Found ACTUAL maze exit at {goal_pos}")
            print(f"🎯 A* will navigate from entrance to exit (matching manual path)")
        
        print(f"🔍 Pathfinding: Start {start_pos}, Goal {goal_pos}")
        
        # Debug: Check what's around start and goal positions
        if start_pos:
            sy, sx = start_pos
            print(f"🚪 Start area tiles: N={self.grid[sy-1, sx]}, S={self.grid[sy+1, sx]}, E={self.grid[sy, sx+1]}, W={self.grid[sy, sx-1]}")
            
        if goal_pos:
            gy, gx = goal_pos
            print(f"👑 Goal area tiles: N={self.grid[gy-1, gx]}, S={self.grid[gy+1, gx]}, E={self.grid[gy, gx+1]}, W={self.grid[gy, gx-1]}")
            
        # Debug: Check if the straight-line path should be blocked
        print(f"🧱 Grid values along straight path:")
        for x in range(60, 70):
            print(f"   (5, {x}) = {self.grid[5, x]}")
        for y in range(6, 19):
            print(f"   ({y}, 69) = {self.grid[y, 69]}")
        
        if not start_pos or not goal_pos:
            print(f"❌ Missing positions: Start={start_pos}, Goal={goal_pos}")
            return []
        
        # A* pathfinding algorithm
        import heapq
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = [(0, start_pos)]
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: heuristic(start_pos, goal_pos)}
        positions_checked = 0
        
        print(f"🚀 A* starting from {start_pos} to {goal_pos}")
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal_pos:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_pos)
                final_path = path[::-1]  # Reverse to get start-to-goal
                print(f"✅ Path found with {len(final_path)} steps: {final_path}")
                
                # Store the A* path for visualization
                self.astar_path = final_path
                self.show_astar_route = False  # Off by default
                print(f"🔴 RED ROUTE: A* optimal path stored for visualization")
                print(f"🎮 Press 'R' to toggle red route visibility")
                
                return final_path
            
            # Check neighbors (up, down, left, right)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dy, current[1] + dx)
                ny, nx = neighbor
                
                # Check bounds and walkable tiles (floor tiles and prison bars) + maze walls
                if (0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH and
                    self.grid[ny, nx] in [0, 3, 4]):  # Floor, prison bars, or queen
                    
                    # Check if movement is blocked using original maze logic
                    if self._is_maze_wall_blocking(current, neighbor):
                        continue  # Skip this neighbor if blocked by maze wall
                    
                    tentative_g = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal_pos)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print(f"❌ A* pathfinding failed - no path found from {start_pos} to {goal_pos}")
        print(f"🔍 Checked {len(came_from)} positions during search")
        return []  # No path found
        
    def _is_blocked_by_maze_wall(self, from_pos, to_pos):
        """Check if movement between two positions is blocked by maze walls."""
        from_y, from_x = from_pos
        to_y, to_x = to_pos
        
        # Only check adjacent moves
        if abs(from_y - to_y) + abs(from_x - to_x) != 1:
            return True  # Block non-adjacent moves
        
        # Debug: Check the problematic straight-line area
        if from_pos == (9, 24) and to_pos == (9, 25):
            print(f"🔍 DEBUG: Checking PROBLEMATIC move from {from_pos} to {to_pos}")
            print(f"🧱 Should be blocked but isn't - checking wall at (25, 9)")
            print(f"🧱 Vertical walls around area: {[(x,y) for (x,y) in self.maze_walls.get('vertical', set()) if 24 <= x <= 26 and 8 <= y <= 10]}")
            print(f"🧱 All vertical walls: {list(self.maze_walls.get('vertical', set()))[:20]}...")
        
        # Check if there's a maze wall blocking this movement
        if hasattr(self, 'maze_walls'):
            # Moving UP (from_y > to_y): check horizontal wall above to_pos
            if to_y < from_y:
                wall_check = (to_x, to_y) in self.maze_walls.get('horizontal', set())
                if from_pos == (5, 59):
                    print(f"🔍 Moving UP: checking wall at ({to_x}, {to_y}) = {wall_check}")
                if wall_check:
                    return True
                    
            # Moving DOWN (from_y < to_y): check horizontal wall below from_pos  
            if to_y > from_y:
                wall_check = (from_x, from_y + 1) in self.maze_walls.get('horizontal', set())
                if from_pos == (5, 59):
                    print(f"🔍 Moving DOWN: checking wall at ({from_x}, {from_y + 1}) = {wall_check}")
                if wall_check:
                    return True
                    
            # Moving LEFT (from_x > to_x): check vertical wall left of from_pos
            if to_x < from_x:
                wall_check = (from_x, from_y) in self.maze_walls.get('vertical', set())
                if from_pos == (5, 59):
                    print(f"🔍 Moving LEFT: checking wall at ({from_x}, {from_y}) = {wall_check}")
                if wall_check:
                    return True
                    
            # Moving RIGHT (from_x < to_x): check vertical wall between cells
            if to_x > from_x:
                wall_check = (to_x, from_y) in self.maze_walls.get('vertical', set())
                if from_pos == (5, 59):
                    print(f"🔍 Moving RIGHT: checking wall at ({to_x}, {from_y}) = {wall_check}")
                if wall_check:
                    return True
        
        return False  # No wall blocking
        
    def _build_blocked_edges(self):
        """Build blocked edges set from maze walls for A* pathfinding."""
        blocked = set()
        
        def add_pair(a, b):
            # Store as ordered tuple - we'll check both directions when testing
            blocked.add((a, b))
        
        # Convert coordinates from (x,y) to (y,x) format for pathfinding
        # Vertical walls block left <-> right movement
        for (wall_x, wall_y) in self.maze_walls['vertical']:
            # Convert to (row, col) format and create the blocked cell pair
            r, c_v = wall_y, wall_x  # Convert (x,y) to (y,x)
            a = (r, c_v - 1)  # Left cell
            b = (r, c_v)      # Right cell
            
            # Validate in-bounds
            if (0 <= a[0] < GRID_HEIGHT and 0 <= a[1] < GRID_WIDTH and 
                0 <= b[0] < GRID_HEIGHT and 0 <= b[1] < GRID_WIDTH):
                add_pair(a, b)
        
        # Horizontal walls block up <-> down movement  
        for (wall_x, wall_y) in self.maze_walls['horizontal']:
            # Convert to (row, col) format and create the blocked cell pair
            r_h, c = wall_y, wall_x  # Convert (x,y) to (y,x)
            a = (r_h - 1, c)  # Upper cell
            b = (r_h, c)      # Lower cell
            
            # Validate in-bounds
            if (0 <= a[0] < GRID_HEIGHT and 0 <= a[1] < GRID_WIDTH and
                0 <= b[0] < GRID_HEIGHT and 0 <= b[1] < GRID_WIDTH):
                add_pair(a, b)
        
        return blocked
        
    def _is_maze_wall_blocking(self, from_pos, to_pos):
        """Check if movement between two adjacent cells is blocked by a maze wall.
        This now consults the authoritative wall sets instead of scanning no_wall_tiles.
        from_pos and to_pos are (y,x).
        """
        fy, fx = from_pos
        ty, tx = to_pos
        # Only adjacent moves are valid in our grid
        if abs(fy - ty) + abs(fx - tx) != 1:
            return True
        return self._is_wall_between((fy, fx), (ty, tx))
        
    def _transfer_maze_walls_to_grid(self):
        """Transfer maze walls from visual representation to grid for pathfinding."""
        # Convert horizontal walls to grid walls
        for wall_x, wall_y in self.maze_walls['horizontal']:
            if (0 <= wall_y < GRID_HEIGHT and 0 <= wall_x < GRID_WIDTH):
                self.grid[wall_y, wall_x] = 1  # Mark as wall
                
        # Convert vertical walls to grid walls  
        for wall_x, wall_y in self.maze_walls['vertical']:
            if (0 <= wall_y < GRID_HEIGHT and 0 <= wall_x < GRID_WIDTH):
                self.grid[wall_y, wall_x] = 1  # Mark as wall
                
        print(f"🧱 Transferred {len(self.maze_walls['horizontal'])} horizontal + {len(self.maze_walls['vertical'])} vertical walls to grid")
        
    def _check_maze_entrance(self, new_pos):
        """Check if player is entering or exiting the maze using precise maze coordinates."""
        y, x = new_pos
        
        # Only check if guard is passed and maze bounds are available
        if not self.guard_passed or not hasattr(self, 'maze_bounds'):
            return
            
        maze_bounds = self.maze_bounds
        
        # Check if player is ENTERING the maze from the right side at entrance_y
        entering_maze_from_right = (x == maze_bounds['right'] and y == maze_bounds['entrance_y'])
        
        # OR check if they're just inside the maze entrance area
        just_inside_maze_from_right = (
            x == maze_bounds['right'] - 1 and 
            maze_bounds['top'] <= y <= maze_bounds['bottom'] and
            not self.step_debug_active
        )
        
        # Check if player is ENTERING the maze from the left side at exit_y (reverse direction)
        entering_maze_from_left = (x == maze_bounds['left'] and y == maze_bounds['exit_y'])
        
        # OR check if they're just inside the maze from left side
        just_inside_maze_from_left = (
            x == maze_bounds['left'] + 1 and 
            maze_bounds['top'] <= y <= maze_bounds['bottom'] and
            not self.step_debug_active
        )
        
        # Combine both entrance directions
        entering_maze = entering_maze_from_right or entering_maze_from_left
        just_inside_maze = just_inside_maze_from_right or just_inside_maze_from_left
        
        # Check if player is EXITING the maze from the left side at exit_y
        exiting_maze_from_left = (
            x == maze_bounds['left'] - 1 and 
            y == maze_bounds['exit_y'] and
            self.step_debug_active
        )
        
        # OR check if they're just outside the left exit area
        just_outside_maze_left = (
            x < maze_bounds['left'] and
            self.step_debug_active
        )
        
        # Check if player is EXITING the maze from the right side at entrance_y (reverse direction)
        exiting_maze_from_right = (
            x == maze_bounds['right'] + 1 and 
            y == maze_bounds['entrance_y'] and
            self.step_debug_active
        )
        
        # OR check if they're just outside the right entrance area
        just_outside_maze_right = (
            x > maze_bounds['right'] and
            self.step_debug_active
        )
        
        # Combine both exit directions
        exiting_maze = exiting_maze_from_left or exiting_maze_from_right
        just_outside_maze = just_outside_maze_left or just_outside_maze_right
        
        if entering_maze or just_inside_maze:
            # Start step tracking
            self.step_debug_active = True
            self.last_position = tuple(new_pos)
            self.step_count = 0
            print(f"")
            print(f"🏛️ MAZE ENTERED! at {new_pos}")
            print(f"🔍 STEP TRACKING ACTIVATED - Recording your navigation path...")
            print(f"📍 Step counting starts from maze entrance coordinates")
            print(f"🎯 Maze bounds: left={maze_bounds['left']}, right={maze_bounds['right']}, entrance_y={maze_bounds['entrance_y']}, exit_y={maze_bounds['exit_y']}")
            print(f"🌫️ MAZE WALLS NOW HIDDEN - Follow your map directions carefully!")
            self.player_in_maze = True
            print(f"")
            
        elif exiting_maze or just_outside_maze:
            # Stop step tracking
            self.step_debug_active = False
            print(f"")
            print(f"🚪 MAZE EXITED! at {new_pos}")
            print(f"🔍 STEP TRACKING DEACTIVATED - Maze navigation complete!")
            print(f"📊 Total steps in maze: {self.step_count}")
            print(f"🎯 Exited at left boundary (exit_y={maze_bounds['exit_y']})")
            print(f"👁️ MAZE WALLS NOW VISIBLE - Well done!")
            self.player_in_maze = False
            print(f"")
            
    def _track_movement_step(self, old_pos, new_pos):
        """Track player movement and output direction."""
        if not self.step_debug_active:
            return
            
        # Calculate direction
        dy = new_pos[0] - old_pos[0]
        dx = new_pos[1] - old_pos[1]
        
        direction_map = {
            (-1, 0): "UP",
            (1, 0): "DOWN", 
            (0, -1): "LEFT",
            (0, 1): "RIGHT"
        }
        
        direction = direction_map.get((dy, dx), "UNKNOWN")
        if direction != "UNKNOWN":
            self.step_count += 1
            print(f"📍 Step {self.step_count}: {direction} (from {old_pos} to {new_pos})")
            
            # Check if reached queen
            if self.grid[new_pos[0], new_pos[1]] == 4:
                print(f"👑 REACHED QUEEN! Total steps: {self.step_count}")
                print(f"🎯 Your actual path can be used to generate proper maze navigation!")
        
    def _generate_inverted_instructions(self, path):
        """Generate inverted navigation instructions from the solution path."""
        if len(path) < 2:
            return []
        
        instructions = []
        direction_map = {
            (-1, 0): "UP",
            (1, 0): "DOWN", 
            (0, -1): "LEFT",
            (0, 1): "RIGHT"
        }
        
        # Inversion mapping
        invert_map = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT", 
            "RIGHT": "LEFT"
        }
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            dy = next_pos[0] - current[0]
            dx = next_pos[1] - current[1]
            
            actual_direction = direction_map.get((dy, dx))
            if actual_direction:
                # Add inverted direction to instructions
                inverted_direction = invert_map[actual_direction]
                instructions.append(inverted_direction)
                # Only print inverted directions when map is collected (will be called from collect function)
        
        return instructions
        
    def _spawn_maze_navigation_map(self):
        """Spawn the maze navigation map after guard password is accepted."""
        if self.maze_map_spawned:
            return
            
        # Generate solution path through maze
        self.maze_solution_path = self._find_maze_solution_path()
        
        if not self.maze_solution_path:
            print("❌ Could not generate maze solution path!")
            print("🔧 Please check maze connectivity or pathfinding logic.")
            return
            
        # Generate inverted instructions from found path
        self.inverted_instructions = self._generate_inverted_instructions(self.maze_solution_path)
        
        # Find random spawn position between maze wall and guard location
        import random
        attempts = 0
        while attempts < 100:
            # Spawn in the vertical empty area between maze right boundary and guard position
            maze_right = self.maze_bounds.get('right', 18) if hasattr(self, 'maze_bounds') else 18
            guard_x = self.guard_pos[1] if self.guard_pos else 23
            
            # Spawn in the narrow vertical area between maze and guard
            spawn_x = random.randint(maze_right + 1, guard_x - 1)  # Between maze wall and guard
            spawn_y = random.randint(5, 15)
            
            if (0 <= spawn_y < GRID_HEIGHT and 0 <= spawn_x < GRID_WIDTH and
                self.grid[spawn_y, spawn_x] == 0):  # Empty floor
                
                # Spawn the map as an item in the passage area
                self.maze_map_pos = (spawn_y, spawn_x)
                self.maze_map_spawned = True
                self.grid[spawn_y, spawn_x] = 13  # Place map object in grid
                
                print(f"🗺️ MAZE NAVIGATION MAP spawned at {self.maze_map_pos}")
                print(f"📋 Map contains {len(self.inverted_instructions) + 1} navigation steps")
                print(f"🔍 Find and collect the PURPLE MAP in the white corridor area!")
                print(f"🎯 MAP STEPS will sync with your manual steps when you enter the maze!")
                print(f"💜 Look for a PURPLE square in the middle passage area!")
                break
            attempts += 1
        
    def _process_guard_answer(self, answer):
        """Process the guard challenge answer."""
        q = self.guard_challenge_questions[self.guard_current_question]
        
        if answer == q['expected_digit']:
            print(f"✅ Correct! Digit #{q['digit_pos'] + 1} is indeed {q['expected_digit']}")
            self.guard_current_question += 1
            
            if self.guard_current_question >= len(self.guard_challenge_questions):
                # Challenge complete
                print("🛡️ GUARD: 'All keys verified! You may pass.'")
                print("🎉 GUARD PASSED! Access to left chamber granted!")
                self.guard_challenge_active = False
                self.guard_passed = True
            else:
                # Next question
                self._show_current_guard_question()
        else:
            print(f"❌ Wrong! Expected {q['expected_digit']}, got {answer}")
            print("🛡️ GUARD: 'Try again!'")
            self._show_current_guard_question()
        
    def handle_input(self, events):
        """Handle player input - WASD movement (single press + continuous hold)."""
        current_time = pygame.time.get_ticks()
        
        # Handle discrete key presses (immediate response)
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self._try_attach_nearby_box()
                elif event.key == pygame.K_m:  # Read navigation map
                    if self.maze_map_collected and self.inverted_instructions:
                        # Toggle map UI popup
                        self.map_ui_active = not self.map_ui_active
                        if self.map_ui_active:
                            print("📋 MAP UI: Opened - Press 'M' to close")
                        else:
                            print("📋 MAP UI: Closed")
                    else:
                        print("❌ You don't have a navigation map! Find and collect it first.")
                elif event.key == pygame.K_r:
                    self._toggle_astar_route()
                elif event.key == pygame.K_t:  # Test key to check win condition
                    print("=== TESTING WIN CONDITION ===")
                    self._check_rgb_order()
                elif event.key == pygame.K_g:  # Test guard challenge
                    print("=== TESTING GUARD CHALLENGE ===")
                    if len(self.collected_keys) >= 3:
                        self._initiate_guard_challenge()
                    else:
                        print("Need 3 keys first!")
                elif self.guard_password_ui_active:
                    # Handle password input
                    self._handle_guard_password_input(event.key)
                elif self.master_key_password_ui_active:
                    # Handle master key password input
                    self._handle_master_key_password_input(event.key)
                elif self.queen_password_ui_active:
                    # Handle queen password input
                    self._handle_queen_password_input(event.key)
                elif event.key == pygame.K_SPACE:
                    # Close info popup if active
                    if self.info_popup_active:
                        self.info_popup_active = False
        if current_time - self.last_move_time >= self.move_delay:
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_w]:
                self._move_player(pygame.K_w)
                self.last_move_time = current_time
            elif keys[pygame.K_s]:
                self._move_player(pygame.K_s)
                self.last_move_time = current_time
            elif keys[pygame.K_a]:
                self._move_player(pygame.K_a)
                self.last_move_time = current_time
            elif keys[pygame.K_d]:
                self._move_player(pygame.K_d)
                self.last_move_time = current_time
                
    def _move_player(self, key):
        """Move player based on key input, moving attached box as well."""
        old_pos = self.player_pos[:]
        new_pos = self.player_pos[:]
        
        if key == pygame.K_w:
            new_pos[0] -= 1  # Move up
        elif key == pygame.K_s:
            new_pos[0] += 1  # Move down
        elif key == pygame.K_a:
            new_pos[1] -= 1  # Move left
        elif key == pygame.K_d:
            new_pos[1] += 1  # Move right
        else:
            return  # Invalid key
            
        # Check for guard challenge before moving
        if self._check_guard_challenge(new_pos):
            return  # Don't move if blocked by guard
            
        # Check if we can move (considering attached box)
        if self._can_move_with_attached_box(new_pos):
            # Check if entering maze area for step tracking
            self._check_maze_entrance(new_pos)
            
            # Track movement steps if in maze area
            self._track_movement_step(old_pos, new_pos)
            
            # Move player
            self.player_pos = new_pos
            
            # Check for maze navigation map collection
            self._collect_navigation_map(new_pos)
            
            # Check for master key interaction
            self._check_master_key_interaction(new_pos)
            
            # Check for queen challenge interaction
            self._check_queen_interaction(new_pos)
            
            # Check for key collection
            self._try_collect_key(new_pos)
            
            # Move attached box if we have one
            if self.attached_box is not None and self.attachment_offset is not None:
                # Calculate new box position
                old_box_pos = (old_pos[0] + self.attachment_offset[0], old_pos[1] + self.attachment_offset[1])
                new_box_pos = (new_pos[0] + self.attachment_offset[0], new_pos[1] + self.attachment_offset[1])
                
                # Update box position in tracking
                if old_box_pos in self.box_positions:
                    box_type = self.box_positions[old_box_pos]
                    del self.box_positions[old_box_pos]
                    self.box_positions[new_box_pos] = box_type
                    
                    # Check for win condition after moving a box
                    self._check_rgb_order()
                    
    def _can_move_with_attached_box(self, new_player_pos):
        """Check if player can move to new position with attached box."""
        # Simple bounds check for player
        if not (0 <= new_player_pos[0] < GRID_HEIGHT and 0 <= new_player_pos[1] < GRID_WIDTH):
            return False
            
        # Check if player would hit a solid wall (including maze walls)
        if self.grid[new_player_pos[0], new_player_pos[1]] == 1:  # Solid wall
            return False
            
        # Check for maze wall collisions
        if self._hits_maze_wall(tuple(self.player_pos), tuple(new_player_pos)):
            return False
            
        # Check if player position conflicts with non-attached boxes
        if tuple(new_player_pos) in self.box_positions:
            # Allow if it's our attached box position
            if (self.attached_box is not None and self.attachment_offset is not None):
                current_box_pos = (self.player_pos[0] + self.attachment_offset[0], self.player_pos[1] + self.attachment_offset[1])
                if tuple(new_player_pos) != current_box_pos:
                    return False
            else:
                return False
            
        # If we have an attached box, check its new position too
        if self.attached_box is not None and self.attachment_offset is not None:
            new_box_y = new_player_pos[0] + self.attachment_offset[0]
            new_box_x = new_player_pos[1] + self.attachment_offset[1]
            new_box_pos = (new_box_y, new_box_x)
            
            # Check box bounds
            if not (0 <= new_box_y < GRID_HEIGHT and 0 <= new_box_x < GRID_WIDTH):
                return False
                
            # Only check for solid walls, not maze walls
            if self.grid[new_box_y, new_box_x] == 1:  # Solid wall only
                return False
                
            # Check if box conflicts with other non-attached boxes
            if new_box_pos in self.box_positions:
                # Allow if it's the current position of our attached box
                current_box_pos = (self.player_pos[0] + self.attachment_offset[0], self.player_pos[1] + self.attachment_offset[1])
                if new_box_pos != current_box_pos:
                    return False
                    
        return True
            
    def _try_attach_nearby_box(self):
        """Attach/detach a box adjacent to the player (F key)."""
        if self.attached_box is not None:
            # If already attached, detach it
            self._release_attached_box()
            return
            
        player_y, player_x = self.player_pos
        
        # Check all 4 directions for boxes
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dy, dx in directions:
            box_pos = (player_y + dy, player_x + dx)
            if box_pos in self.box_positions:
                # Attach this box
                self.attached_box = self.box_positions[box_pos]
                self.attachment_offset = (dy, dx)  # Store relative position
                
                print(f"Attached {['', '', '', '', '', '', 'Red', 'Green', 'Blue'][self.attached_box]} box")
                return True
        
        print("No box found to attach")
        return False
        
    def _release_attached_box(self):
        """Release the attached box (F key)."""
        if self.attached_box is None:
            print("No box attached")
            return False
            
        print(f"Released {['', '', '', '', '', '', 'Red', 'Green', 'Blue'][self.attached_box]} box")
        self.attached_box = None
        self.attachment_offset = None
        
        # Check for win condition after releasing a box
        self._check_rgb_order()
        return True
                
    def _try_push_box(self, box_pos, direction):
        """Try to push a box in the given direction."""
        box_y, box_x = box_pos
        dy, dx = direction
        
        # Calculate where the box would go
        new_box_y = box_y + dy
        new_box_x = box_x + dx
        new_box_pos = (new_box_y, new_box_x)
        
        # Check if the new position is valid for the box
        if (0 <= new_box_y < GRID_HEIGHT and 0 <= new_box_x < GRID_WIDTH and
            self.grid[new_box_y, new_box_x] == 0 and  # Target is floor
            new_box_pos not in self.box_positions):  # No other box there
            
            # Move the box
            box_type = self.box_positions[box_pos]
            del self.box_positions[box_pos]  # Remove from old position
            self.box_positions[new_box_pos] = box_type  # Add to new position
            
            # Update grid
            self.grid[box_y, box_x] = 0  # Clear old position
            self.grid[new_box_y, new_box_x] = box_type  # Place in new position
            
            # If the box was in a slot area, restore the background
            if self._is_slot_position(box_y, box_x):
                # Check if it's directly under a slot (y=1 under wall slot at y=0)
                if box_y == 1:  # Box was in slot area
                    self.grid[box_y, box_x] = 0  # Just floor (slot is in wall above)
            
            print(f"Pushed {['', '', '', '', '', '', 'Red', 'Green', 'Blue'][box_type]} box from {box_pos} to {new_box_pos}")
            return True
        
        return False
        
    def _is_slot_position(self, y, x):
        """Check if position is a key slot."""
        # Slot positions are at y=1 and specific x positions in right chamber
        return y == 1 and 62 <= x <= 75  # Approximate slot area
            
    def _is_valid_move(self, pos):
        """Check if move is valid (including maze wall collisions)."""
        y, x = pos
        
        # Check bounds
        if not (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH):
            return False
        
        # Check if position is wall
        if self.grid[y, x] == 1:
            return False
            
        # Check maze wall collisions
        if self._hits_maze_wall(self.player_pos, pos):
            return False
            
        return True
        
    def _hits_maze_wall(self, old_pos, new_pos):
        """Check if movement from old_pos to new_pos crosses a maze wall."""
        old_y, old_x = old_pos
        new_y, new_x = new_pos
        
        # Moving right - check vertical wall
        if new_x > old_x:
            wall_pos = (new_x, old_y)
            if wall_pos in self.maze_walls['vertical']:
                return True
        
        # Moving left - check vertical wall
        elif new_x < old_x:
            wall_pos = (old_x, old_y)
            if wall_pos in self.maze_walls['vertical']:
                return True
        
        # Moving down - check horizontal wall
        elif new_y > old_y:
            wall_pos = (old_x, new_y)
            if wall_pos in self.maze_walls['horizontal']:
                return True
        
        # Moving up - check horizontal wall
        elif new_y < old_y:
            wall_pos = (old_x, old_y)
            if wall_pos in self.maze_walls['horizontal']:
                return True
        
        return False
        
    def render(self):
        """Render the game."""
        self.screen.fill(BLACK)
        
        # Update camera to follow player
        self.camera.update(self.player_pos[1], self.player_pos[0])
        
        # Draw grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                screen_x = x * TILE_SIZE - self.camera.x
                screen_y = y * TILE_SIZE - self.camera.y
                
                # Only draw tiles that are visible on screen
                if (-TILE_SIZE <= screen_x <= WINDOW_WIDTH and 
                    -TILE_SIZE <= screen_y <= WINDOW_HEIGHT):
                    
                    # Choose color based on tile type
                    if self.grid[y, x] == 1:  # Wall
                        color = GRAY
                    elif self.grid[y, x] == 3:  # Prison bars
                        color = DARK_GRAY
                    elif self.grid[y, x] == 4:  # Queen
                        color = PURPLE
                    elif self.grid[y, x] == 5:  # Key slot
                        color = GOLD
                    elif self.grid[y, x] == 6:  # Red box
                        color = RED
                    elif self.grid[y, x] == 7:  # Green box
                        color = GREEN
                    elif self.grid[y, x] == 8:  # Blue box
                        color = BLUE
                    elif self.grid[y, x] == 9:  # Red key
                        color = ORANGE  # Distinct from red box
                    elif self.grid[y, x] == 10:  # Green key  
                        color = CYAN    # Distinct from green box
                    elif self.grid[y, x] == 11:  # Blue key
                        color = YELLOW  # Distinct from blue box
                    elif self.grid[y, x] == 12:  # Guard NPC
                        color = BROWN  # Brown for guard
                    elif self.grid[y, x] == 13:  # Navigation map
                        color = PURPLE  # Purple for map item
                    elif self.grid[y, x] == 14:  # Master key
                        color = GOLD  # Gold for master key
                    else:  # Floor
                        color = WHITE
                    
                    # Draw tile
                    rect = pygame.Rect(screen_x, screen_y, TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    
                    # If this is a slot position, check if there's a box on top and draw both
                    if self.grid[y, x] == 5:  # This is a slot
                        if (y, x) in self.box_positions:
                            # Draw the colored box on top of the gold slot
                            box_type = self.box_positions[(y, x)]
                            if box_type == 6:
                                box_color = RED
                            elif box_type == 7:
                                box_color = GREEN
                            elif box_type == 8:
                                box_color = BLUE
                            else:
                                box_color = WHITE
                            
                            # Draw box on top of slot
                            pygame.draw.rect(self.screen, box_color, rect)
        
        # Draw maze walls as thin lines
        self._draw_maze_walls()
        
        # Draw A* red route if enabled
        if self.show_astar_route and self.astar_path:
            self._draw_astar_route()
        
        # Draw player
        player_screen_x = self.player_pos[1] * TILE_SIZE - self.camera.x
        player_screen_y = self.player_pos[0] * TILE_SIZE - self.camera.y
        player_rect = pygame.Rect(player_screen_x, player_screen_y, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(self.screen, PLAYER_BLUE, player_rect)
        pygame.draw.rect(self.screen, WHITE, player_rect, 2)
        
        # Draw boxes (they render as full separate tiles now)
        for (box_y, box_x), box_type in self.box_positions.items():
            box_screen_x = box_x * TILE_SIZE - self.camera.x
            box_screen_y = box_y * TILE_SIZE - self.camera.y
            
            # Only draw if visible on screen
            if (-TILE_SIZE <= box_screen_x <= WINDOW_WIDTH and 
                -TILE_SIZE <= box_screen_y <= WINDOW_HEIGHT):
                
                if box_type == 6:
                    box_color = RED
                elif box_type == 7:
                    box_color = GREEN
                elif box_type == 8:
                    box_color = BLUE
                else:
                    box_color = WHITE
                
                box_rect = pygame.Rect(box_screen_x, box_screen_y, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, box_color, box_rect)
                
                # Add a border to distinguish attached boxes
                if (self.attached_box is not None and 
                    self.attachment_offset is not None and
                    (box_y, box_x) == (self.player_pos[0] + self.attachment_offset[0], 
                                       self.player_pos[1] + self.attachment_offset[1])):
                    pygame.draw.rect(self.screen, WHITE, box_rect, 3)  # Thick white border for attached box
        
        # Draw key objects as small circles
        for (key_y, key_x), key_data in self.key_objects.items():
            key_screen_x = key_x * TILE_SIZE - self.camera.x
            key_screen_y = key_y * TILE_SIZE - self.camera.y
            
            # Only draw if visible on screen
            if (-TILE_SIZE <= key_screen_x <= WINDOW_WIDTH and 
                -TILE_SIZE <= key_screen_y <= WINDOW_HEIGHT):
                
                # Determine key color
                key_type = key_data['type']
                if key_type == 9:  # Red key
                    key_color = ORANGE
                elif key_type == 10:  # Green key
                    key_color = CYAN
                elif key_type == 11:  # Blue key
                    key_color = YELLOW
                else:
                    key_color = WHITE
                
                # Draw small circle in center of tile
                circle_center = (key_screen_x + TILE_SIZE // 2, key_screen_y + TILE_SIZE // 2)
                circle_radius = TILE_SIZE // 4  # Quarter of tile size
                pygame.draw.circle(self.screen, key_color, circle_center, circle_radius)
                pygame.draw.circle(self.screen, WHITE, circle_center, circle_radius, 2)  # White border
        
        # Draw UI info
        self._draw_ui()
        
        # Draw password popup if active
        if self.guard_password_ui_active:
            self._draw_password_popup()
            
        # Draw master key password popup if active
        if self.master_key_password_ui_active:
            self._draw_master_key_password_popup()
            
        # Draw queen password popup if active
        if self.queen_password_ui_active:
            self._draw_queen_password_popup()
            
        # Draw info popup if active (on top of everything)
        if self.info_popup_active:
            self._draw_info_popup()
            
        # Draw map reading UI if active
        if self.map_ui_active:
            self._draw_map_reading_ui()
            
    def _draw_debug_grid(self):
        """Draw grid lines for debugging."""
        grid_color = (100, 100, 100)  # Dark gray
        
        # Draw vertical grid lines
        for x in range(0, GRID_WIDTH + 1):
            screen_x = x * TILE_SIZE - self.camera.x
            if -10 <= screen_x <= WINDOW_WIDTH + 10:  # Only draw visible lines
                start_pos = (screen_x, -self.camera.y)
                end_pos = (screen_x, GRID_HEIGHT * TILE_SIZE - self.camera.y)
                pygame.draw.line(self.screen, grid_color, start_pos, end_pos, 1)
        
        # Draw horizontal grid lines  
        for y in range(0, GRID_HEIGHT + 1):
            screen_y = y * TILE_SIZE - self.camera.y
            if -10 <= screen_y <= WINDOW_HEIGHT + 10:  # Only draw visible lines
                start_pos = (-self.camera.x, screen_y)
                end_pos = (GRID_WIDTH * TILE_SIZE - self.camera.x, screen_y)
                pygame.draw.line(self.screen, grid_color, start_pos, end_pos, 1)
        
    def _draw_maze_walls(self):
        """Draw thin maze walls as lines."""
        # Hide maze walls when player is inside for challenging gameplay
        if self.player_in_maze:
            return  # Don't draw maze walls when player is navigating inside
            
        line_width = 2
        
        # Draw horizontal walls
        for wall_x, wall_y in self.maze_walls['horizontal']:
            screen_x = wall_x * TILE_SIZE - self.camera.x
            screen_y = wall_y * TILE_SIZE - self.camera.y
            
            # Only draw if visible on screen
            if (-TILE_SIZE <= screen_x <= WINDOW_WIDTH and 
                -TILE_SIZE <= screen_y <= WINDOW_HEIGHT):
                
                start_pos = (screen_x, screen_y)
                end_pos = (screen_x + TILE_SIZE, screen_y)
                pygame.draw.line(self.screen, BLACK, start_pos, end_pos, line_width)
        
        # Draw vertical walls
        for wall_x, wall_y in self.maze_walls['vertical']:
            screen_x = wall_x * TILE_SIZE - self.camera.x
            screen_y = wall_y * TILE_SIZE - self.camera.y
            
            # Only draw if visible on screen
            if (-TILE_SIZE <= screen_x <= WINDOW_WIDTH and 
                -TILE_SIZE <= screen_y <= WINDOW_HEIGHT):
                
                start_pos = (screen_x, screen_y)
                end_pos = (screen_x, screen_y + TILE_SIZE)
                pygame.draw.line(self.screen, BLACK, start_pos, end_pos, line_width)
        
    def _draw_astar_route(self):
        """Draw the A* optimal path as red squares with connecting lines."""
        if not self.astar_path:
            return
            
        route_color = RED
        line_width = 4
        
        # Draw path squares and connecting lines
        for i, (path_y, path_x) in enumerate(self.astar_path):
            screen_x = path_x * TILE_SIZE - self.camera.x
            screen_y = path_y * TILE_SIZE - self.camera.y
            
            # Only draw if visible on screen
            if (-TILE_SIZE <= screen_x <= WINDOW_WIDTH and 
                -TILE_SIZE <= screen_y <= WINDOW_HEIGHT):
                
                # Draw semi-transparent red square for path step
                path_rect = pygame.Rect(screen_x + 4, screen_y + 4, TILE_SIZE - 8, TILE_SIZE - 8)
                pygame.draw.rect(self.screen, route_color, path_rect, 3)
                
                # Draw step number
                if i < len(self.astar_path) - 1:  # Don't draw number on last step
                    step_text = str(i + 1)
                    step_surface = self.font.render(step_text, True, WHITE)
                    text_x = screen_x + (TILE_SIZE - step_surface.get_width()) // 2
                    text_y = screen_y + (TILE_SIZE - step_surface.get_height()) // 2
                    self.screen.blit(step_surface, (text_x, text_y))
                
                # Draw connecting line to next step
                if i < len(self.astar_path) - 1:
                    next_y, next_x = self.astar_path[i + 1]
                    next_screen_x = next_x * TILE_SIZE - self.camera.x
                    next_screen_y = next_y * TILE_SIZE - self.camera.y
                    
                    # Draw line from center of current tile to center of next tile
                    start_center = (screen_x + TILE_SIZE // 2, screen_y + TILE_SIZE // 2)
                    end_center = (next_screen_x + TILE_SIZE // 2, next_screen_y + TILE_SIZE // 2)
                    pygame.draw.line(self.screen, route_color, start_center, end_center, line_width)
        
    def _draw_ui(self):
        """Draw UI information."""
        # Player position
        pos_text = f"Position: ({self.player_pos[1]}, {self.player_pos[0]})"
        pos_surface = self.font.render(pos_text, True, WHITE)
        self.screen.blit(pos_surface, (10, 10))
        
        # Grid info
        grid_text = f"Map: {GRID_WIDTH}x{GRID_HEIGHT}"
        grid_surface = self.font.render(grid_text, True, WHITE)
        self.screen.blit(grid_surface, (10, 40))
        
        # Key count
        inventory_text = f"Keys: {len(self.collected_keys)}/3"
        inventory_surface = self.font.render(inventory_text, True, WHITE)
        self.screen.blit(inventory_surface, (10, 70))
        
        # Camera info
        cam_text = f"Camera: ({int(self.camera.x)}, {int(self.camera.y)})"
        cam_surface = self.font.render(cam_text, True, WHITE)
        self.screen.blit(cam_surface, (10, 100))
        
        # Show key values temporarily when collected
        current_time = pygame.time.get_ticks()
        if (self.key_display_timer > 0 and 
            current_time - self.key_display_timer < self.key_display_duration and 
            self.collected_keys):
            
            y_offset = 130
            for key in self.collected_keys:
                key_text = f"{key['color']}: {key['value']}"
                key_surface = self.font.render(key_text, True, YELLOW)
                self.screen.blit(key_surface, (10, y_offset))
                y_offset += 25
                
    def _draw_password_popup(self):
        """Draw the password entry popup UI."""
        # Create smaller font for better fit
        small_font = pygame.font.Font(None, 18)
        
        # Create popup background - larger to fit challenge info
        popup_width = 550
        popup_height = 400
        popup_x = (WINDOW_WIDTH - popup_width) // 2
        popup_y = (WINDOW_HEIGHT - popup_height) // 2
        
        # Background with border
        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
        pygame.draw.rect(self.screen, DARK_GRAY, popup_rect)
        pygame.draw.rect(self.screen, GOLD, popup_rect, 3)
        
        # Title
        title_text = "🛡️ GUARD PASSWORD VERIFICATION"
        title_surface = small_font.render(title_text, True, GOLD)
        title_x = popup_x + (popup_width - title_surface.get_width()) // 2
        self.screen.blit(title_surface, (title_x, popup_y + 15))
        
        # Show digit challenges
        y_pos = popup_y + 50
        
        instruction_text = "Selected digits from your RGB keys:"
        instruction_surface = small_font.render(instruction_text, True, WHITE)
        inst_x = popup_x + (popup_width - instruction_surface.get_width()) // 2
        self.screen.blit(instruction_surface, (inst_x, y_pos))
        y_pos += 30
        
        # Show the specific digit positions being used
        if hasattr(self, 'master_key_digit_info') and self.master_key_digit_info:
            for info in self.master_key_digit_info:
                digit_text = f"{info['color']} Key: {self._ordinal(info['position'])} digit = {info['digit']}"
                digit_surface = small_font.render(digit_text, True, CYAN)
                digit_x = popup_x + 30
                self.screen.blit(digit_surface, (digit_x, y_pos))
                y_pos += 25
        
        y_pos += 10
            
        if hasattr(self, 'guard_digit_challenges') and self.guard_digit_challenges:
            instruction_text = "Enter these digits in order:"
            instruction_surface = small_font.render(instruction_text, True, WHITE)
            inst_x = popup_x + (popup_width - instruction_surface.get_width()) // 2
            self.screen.blit(instruction_surface, (inst_x, y_pos))
            y_pos += 30
            
            # Show each challenge
            for challenge in self.guard_digit_challenges:
                ordinal = self._ordinal(challenge['digit_position'])
                challenge_text = f"{challenge['challenge_num']}. {ordinal} digit of {challenge['color']} key"
                challenge_surface = small_font.render(challenge_text, True, CYAN)
                challenge_x = popup_x + 30
                self.screen.blit(challenge_surface, (challenge_x, y_pos))
                y_pos += 25
        else:
            # Fallback if no challenges available
            instruction_text = "Enter Password"
            instruction_surface = small_font.render(instruction_text, True, WHITE)
            inst_x = popup_x + (popup_width - instruction_surface.get_width()) // 2
            self.screen.blit(instruction_surface, (inst_x, y_pos))
            y_pos += 40
            
        # Enter password instruction
        enter_text = "Combine all answers as 5-digit password:"
        enter_surface = small_font.render(enter_text, True, WHITE)
        enter_x = popup_x + (popup_width - enter_surface.get_width()) // 2
        self.screen.blit(enter_surface, (enter_x, y_pos + 20))
        
        # Input box
        input_box_rect = pygame.Rect(popup_x + 75, popup_y + 280, popup_width - 150, 35)
        pygame.draw.rect(self.screen, WHITE, input_box_rect)
        pygame.draw.rect(self.screen, BLACK, input_box_rect, 2)
        
        # Current input text
        input_text = self.guard_password_input + "_"  # Show cursor
        input_surface = small_font.render(input_text, True, BLACK)
        text_x = input_box_rect.x + (input_box_rect.width - input_surface.get_width()) // 2
        self.screen.blit(input_surface, (text_x, input_box_rect.y + 10))
        
        # Help text
        help_text = "Press ENTER to submit • ESC to cancel"
        help_surface = small_font.render(help_text, True, (180, 180, 180))
        help_x = popup_x + (popup_width - help_surface.get_width()) // 2
        self.screen.blit(help_surface, (help_x, popup_y + 330))
        
    def _draw_master_key_password_popup(self):
        """Draw the master key password entry popup UI."""
        # Create smaller font for better fit
        small_font = pygame.font.Font(None, 20)
        
        # Create popup background - smaller size
        popup_width = 500
        popup_height = 300
        popup_x = (WINDOW_WIDTH - popup_width) // 2
        popup_y = (WINDOW_HEIGHT - popup_height) // 2
        
        # Background with border
        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
        pygame.draw.rect(self.screen, (30, 30, 40), popup_rect)  # Dark background
        pygame.draw.rect(self.screen, GOLD, popup_rect, 4)
        
        # Inner accent border
        inner_rect = pygame.Rect(popup_x + 3, popup_y + 3, popup_width - 6, popup_height - 6)
        pygame.draw.rect(self.screen, (200, 150, 0), inner_rect, 2)
        
        # Title - smaller font
        title_text = "🗝️ MASTER KEY CHALLENGE"
        title_surface = small_font.render(title_text, True, GOLD)
        title_x = popup_x + (popup_width - title_surface.get_width()) // 2
        self.screen.blit(title_surface, (title_x, popup_y + 15))
        
        # Show digit challenges
        y_pos = popup_y + 50
        
        instruction_text = "Selected digits from your RGB keys:"
        instruction_surface = small_font.render(instruction_text, True, WHITE)
        inst_x = popup_x + (popup_width - instruction_surface.get_width()) // 2
        self.screen.blit(instruction_surface, (inst_x, y_pos))
        y_pos += 30
        
        # Show the specific digit positions being used (WITHOUT the actual values)
        if hasattr(self, 'master_key_digit_info') and self.master_key_digit_info:
            for info in self.master_key_digit_info:
                digit_text = f"{info['color']} Key: Use {self._ordinal(info['position'])} digit"
                digit_surface = small_font.render(digit_text, True, CYAN)
                digit_x = popup_x + 30
                self.screen.blit(digit_surface, (digit_x, y_pos))
                y_pos += 25
        
        y_pos += 10
        
        # Show proper instruction without revealing digits
        instruction_text = self.master_key_instruction_text
        instruction_surface = small_font.render(instruction_text, True, WHITE)
        inst_x = popup_x + (popup_width - instruction_surface.get_width()) // 2
        self.screen.blit(instruction_surface, (inst_x, popup_y + 70))
        
        # Enter result instruction
        enter_text = "Enter result (1-5 digits):"
        enter_surface = small_font.render(enter_text, True, WHITE)
        enter_x = popup_x + (popup_width - enter_surface.get_width()) // 2
        self.screen.blit(enter_surface, (enter_x, popup_y + 100))
        
        # Input box - smaller
        input_box_rect = pygame.Rect(popup_x + 175, popup_y + 125, 150, 30)
        pygame.draw.rect(self.screen, WHITE, input_box_rect)
        pygame.draw.rect(self.screen, BLACK, input_box_rect, 2)
        
        # Current input text
        input_text = self.master_key_password_input + "_"  # Show cursor
        input_surface = small_font.render(input_text, True, BLACK)
        text_x = input_box_rect.x + (input_box_rect.width - input_surface.get_width()) // 2
        self.screen.blit(input_surface, (text_x, input_box_rect.y + 8))
        
        # Help text - smaller font
        help_text = "Press ENTER to submit • ESC to cancel"
        help_surface = small_font.render(help_text, True, (180, 180, 180))
        help_x = popup_x + (popup_width - help_surface.get_width()) // 2
        self.screen.blit(help_surface, (help_x, popup_y + 180))
        
    def _draw_queen_password_popup(self):
        """Draw the queen password entry popup UI."""
        # Create smaller font for better fit
        small_font = pygame.font.Font(None, 20)
        
        # Create popup background
        popup_width = 550
        popup_height = 350
        popup_x = (WINDOW_WIDTH - popup_width) // 2
        popup_y = (WINDOW_HEIGHT - popup_height) // 2
        
        # Background with border - royal purple theme
        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
        pygame.draw.rect(self.screen, (40, 20, 60), popup_rect)  # Dark purple background
        pygame.draw.rect(self.screen, (200, 100, 255), popup_rect, 4)  # Purple border
        
        # Inner accent border
        inner_rect = pygame.Rect(popup_x + 3, popup_y + 3, popup_width - 6, popup_height - 6)
        pygame.draw.rect(self.screen, (150, 75, 200), inner_rect, 2)
        
        # Title - royal style
        title_text = "👑 QUEEN'S FINAL CHALLENGE"
        title_surface = small_font.render(title_text, True, (255, 215, 0))  # Gold
        title_x = popup_x + (popup_width - title_surface.get_width()) // 2
        self.screen.blit(title_surface, (title_x, popup_y + 15))
        
        # Challenge description
        desc_y = popup_y + 50
        
        # Show the specific formula (WITHOUT revealing the actual values)
        if hasattr(self, 'queen_chosen_digit_position'):
            desc_texts = [
                "Final Password Formula:",
                f"{self._ordinal(self.queen_chosen_digit_position)} digit of Master Key + Reversed Guard Password",
                f"• Use the {self._ordinal(self.queen_chosen_digit_position)} digit from your Master Key result",
                f"• Reverse your Guard Password (5 digits)",
                f"• Combine: [digit] + [reversed password]",
                f"Attempts remaining: {3 - self.queen_attempts}/3"
            ]
        else:
            desc_texts = [
                "Final Password Formula:",
                "Specific digit from Master Key + Reversed Guard Password",
                f"Attempts remaining: {3 - self.queen_attempts}/3"
            ]
        
        for i, desc_text in enumerate(desc_texts):
            color = (255, 255, 255) if i != 2 else (255, 100, 100)  # Red for attempts
            desc_surface = small_font.render(desc_text, True, color)
            desc_x = popup_x + (popup_width - desc_surface.get_width()) // 2
            self.screen.blit(desc_surface, (desc_x, desc_y + i * 25))
        
        # Enter password instruction
        enter_text = "Enter the combined password:"
        enter_surface = small_font.render(enter_text, True, (255, 255, 255))
        enter_x = popup_x + (popup_width - enter_surface.get_width()) // 2
        self.screen.blit(enter_surface, (enter_x, popup_y + 150))
        
        # Input box - royal styling
        input_box_rect = pygame.Rect(popup_x + 125, popup_y + 180, 300, 40)
        pygame.draw.rect(self.screen, (255, 255, 255), input_box_rect)
        pygame.draw.rect(self.screen, (200, 100, 255), input_box_rect, 3)
        
        # Current input text
        input_text = self.queen_password_input + "_"  # Show cursor
        input_surface = small_font.render(input_text, True, (0, 0, 0))
        text_x = input_box_rect.x + (input_box_rect.width - input_surface.get_width()) // 2
        self.screen.blit(input_surface, (text_x, input_box_rect.y + 12))
        
        # Warning text
        warning_text = "⚠️ FINAL CHALLENGE - 3 attempts only!"
        warning_surface = small_font.render(warning_text, True, (255, 100, 100))
        warning_x = popup_x + (popup_width - warning_surface.get_width()) // 2
        self.screen.blit(warning_surface, (warning_x, popup_y + 240))
        
        # Help text
        help_text = "Press ENTER to submit • ESC to cancel"
        help_surface = small_font.render(help_text, True, (180, 180, 180))
        help_x = popup_x + (popup_width - help_surface.get_width()) // 2
        self.screen.blit(help_surface, (help_x, popup_y + 280))
        
    def _show_info_popup(self, title, messages, auto_close_seconds=0):
        """Show an informational popup with title and messages."""
        self.info_popup_active = True
        self.info_popup_title = title
        self.info_popup_messages = messages if isinstance(messages, list) else [messages]
        
        if auto_close_seconds > 0:
            self.info_popup_auto_close_time = pygame.time.get_ticks() + (auto_close_seconds * 1000)
        else:
            self.info_popup_auto_close_time = 0
            
    def _show_proximity_popup(self, title, messages, trigger_pos, distance=2):
        """Show a proximity-based popup that appears when near and disappears when far."""
        self.info_popup_active = True
        self.info_popup_title = title
        self.info_popup_messages = messages if isinstance(messages, list) else [messages]
        self.info_popup_auto_close_time = 0  # No auto-close for proximity popups
        self.info_popup_trigger_pos = trigger_pos
        self.info_popup_trigger_distance = distance
        
    def _update_proximity_popups(self):
        """Update proximity-based popups based on player position."""
        if not self.info_popup_active or not self.info_popup_trigger_pos:
            return
            
        # Calculate distance from player to trigger position
        player_y, player_x = self.player_pos
        trigger_y, trigger_x = self.info_popup_trigger_pos
        distance = abs(player_y - trigger_y) + abs(player_x - trigger_x)
        
        # Close popup if player moved too far away
        if distance > self.info_popup_trigger_distance:
            self.info_popup_active = False
            self.info_popup_trigger_pos = None
            
    def _draw_info_popup(self):
        """Draw the informational popup UI."""
        if not self.info_popup_active:
            return
            
        # Auto-close if time is up
        if self.info_popup_auto_close_time > 0 and pygame.time.get_ticks() >= self.info_popup_auto_close_time:
            self.info_popup_active = False
            return
            
        # Create smaller font for better fit
        small_font = pygame.font.Font(None, 18)
        title_font = pygame.font.Font(None, 22)
        
        # Create popup background - dynamic size based on content
        popup_width = 600
        popup_height = 100 + len(self.info_popup_messages) * 25  # Dynamic height
        popup_x = (WINDOW_WIDTH - popup_width) // 2
        popup_y = (WINDOW_HEIGHT - popup_height) // 2
        
        # Background with border - info theme (blue)
        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
        pygame.draw.rect(self.screen, (20, 40, 60), popup_rect)  # Dark blue background
        pygame.draw.rect(self.screen, (100, 150, 255), popup_rect, 4)  # Blue border
        
        # Inner accent border
        inner_rect = pygame.Rect(popup_x + 3, popup_y + 3, popup_width - 6, popup_height - 6)
        pygame.draw.rect(self.screen, (75, 125, 200), inner_rect, 2)
        
        # Title - info style
        title_surface = title_font.render(self.info_popup_title, True, (255, 255, 100))  # Yellow
        title_x = popup_x + (popup_width - title_surface.get_width()) // 2
        self.screen.blit(title_surface, (title_x, popup_y + 15))
        
        # Messages
        y_offset = popup_y + 50
        for message in self.info_popup_messages:
            message_surface = small_font.render(message, True, (255, 255, 255))
            message_x = popup_x + (popup_width - message_surface.get_width()) // 2
            self.screen.blit(message_surface, (message_x, y_offset))
            y_offset += 25
            
        # Help text
        help_text = "Press SPACE to close" if self.info_popup_auto_close_time == 0 else f"Auto-closes in {max(0, (self.info_popup_auto_close_time - pygame.time.get_ticks()) // 1000 + 1)} seconds"
        help_surface = small_font.render(help_text, True, (180, 180, 180))
        help_x = popup_x + (popup_width - help_surface.get_width()) // 2
        self.screen.blit(help_surface, (help_x, popup_y + popup_height - 30))
        
    def _draw_map_reading_ui(self):
        """Draw the beautiful navigation map reading UI with dynamic sizing."""
        # Create paragraph text from directions
        directions_text = ", ".join(self.inverted_instructions)
        
        # Calculate dynamic window size based on content
        padding = 60  # Increased padding
        line_height = 32
        
        # Calculate maximum available width (leave margin for screen boundaries)
        max_available_width = WINDOW_WIDTH - 120  # 60px margin on each side
        
        # Header text
        header_text = "ANCIENT MAP - DIRECTIONS ARE INVERTED"
        header_surface = self.font.render(header_text, True, YELLOW)
        header_width = header_surface.get_width()
        
        # Smart text wrapping based on actual pixel width, not character count
        words = directions_text.split(", ")
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + ", " if current_line else word + ", "
            test_surface = self.font.render(test_line.rstrip(", "), True, CYAN)
            
            # Check if this line would fit within our width constraint
            if test_surface.get_width() <= max_available_width - padding * 2:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.rstrip(", "))
                current_line = word + ", "
        
        if current_line:
            lines.append(current_line.rstrip(", "))
        
        # Calculate actual required width based on rendered text
        max_line_width = header_width
        for line in lines:
            line_surface = self.font.render(line, True, CYAN)
            max_line_width = max(max_line_width, line_surface.get_width())
        
        # Dynamic sizing with proper constraints
        popup_width = min(max_line_width + padding * 2, max_available_width)
        popup_height = 140 + len(lines) * line_height + 60  # Header + content + bottom padding
        
        # Ensure minimum size
        popup_width = max(500, popup_width)
        popup_height = max(200, popup_height)
        
        # Center the popup
        popup_x = (WINDOW_WIDTH - popup_width) // 2
        popup_y = (WINDOW_HEIGHT - popup_height) // 2
        
        # Create beautiful gradient background
        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
        
        # Dark background with subtle gradient effect
        pygame.draw.rect(self.screen, (40, 40, 50), popup_rect)  # Darker base
        pygame.draw.rect(self.screen, (50, 50, 60), popup_rect, 0)  # Main background
        
        # Beautiful border with multiple layers
        pygame.draw.rect(self.screen, PURPLE, popup_rect, 4)  # Outer border
        inner_rect = pygame.Rect(popup_x + 2, popup_y + 2, popup_width - 4, popup_height - 4)
        pygame.draw.rect(self.screen, (100, 50, 100), inner_rect, 2)  # Inner accent
        
        # Header with better styling
        header_y = popup_y + 25
        header_x = popup_x + (popup_width - header_surface.get_width()) // 2
        
        # Header shadow effect
        shadow_surface = self.font.render(header_text, True, (20, 20, 20))
        self.screen.blit(shadow_surface, (header_x + 2, header_y + 2))
        # Main header
        self.screen.blit(header_surface, (header_x, header_y))
        
        # Decorative line under header
        line_y = header_y + 40
        pygame.draw.line(self.screen, PURPLE, 
                        (popup_x + 30, line_y), 
                        (popup_x + popup_width - 30, line_y), 2)
        
        # Display directions with better formatting
        content_start_y = line_y + 20
        for i, line in enumerate(lines):
            line_surface = self.font.render(line, True, CYAN)
            
            # Center each line
            line_x = popup_x + (popup_width - line_surface.get_width()) // 2
            line_y = content_start_y + i * line_height
            
            # Add subtle text shadow
            shadow_surface = self.font.render(line, True, (10, 10, 10))
            self.screen.blit(shadow_surface, (line_x + 1, line_y + 1))
            # Main text
            self.screen.blit(line_surface, (line_x, line_y))
        
        # Bottom instruction with better styling
        controls_text = "Press M to close"
        controls_surface = self.font.render(controls_text, True, (200, 200, 100))
        controls_x = popup_x + (popup_width - controls_surface.get_width()) // 2
        controls_y = popup_y + popup_height - 30
        
        # Controls shadow
        controls_shadow = self.font.render(controls_text, True, (20, 20, 20))
        self.screen.blit(controls_shadow, (controls_x + 1, controls_y + 1))
        self.screen.blit(controls_surface, (controls_x, controls_y))
        
    def _toggle_map_ui(self):
        """Toggle the navigation map UI display."""
        self.map_ui_active = not self.map_ui_active
        if self.map_ui_active:
            print("📋 MAP UI: Opened - Press 'M' to close")
        else:
            print("📋 MAP UI: Closed")
            
    def _toggle_astar_route(self):
        """Toggle the A* red route visualization."""
        if not self.maze_map_collected:
            print("❌ RED ROUTE: Must collect the maze map first!")
            print("🗺️ Find and collect the PURPLE map to unlock red route visibility")
            return
            
        if not self.astar_path:
            print("❌ No A* route available - complete challenges first!")
            return
            
        self.show_astar_route = not self.show_astar_route
        if self.show_astar_route:
            print(f"🔴 RED ROUTE: ON - Showing A* optimal path ({len(self.astar_path)} steps)")
        else:
            print("🔴 RED ROUTE: OFF - A* path hidden")
            
    def run(self):
        """Main game loop."""
        while self.running:
            # Handle events
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Update
            self.handle_input(events)
            self._update_proximity_popups()  # Update proximity-based popups
            
            # Render
            self.render()
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()

if __name__ == "__main__":
    game = SimpleDungeon()
    game.run()
