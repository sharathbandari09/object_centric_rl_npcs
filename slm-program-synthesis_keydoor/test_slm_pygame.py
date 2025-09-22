#!/usr/bin/env python3
"""
Test SLM KeyDoor Agent with pygame visualization
"""

import pygame
import numpy as np
import sys
import time
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from keydoor_slm_env import KeyDoorSLMEnv
from slm_agent import SLMKeyDoorAgent

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Pygame setup
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FLOOR_COLOR = (240, 240, 240)
WALL_COLOR = (100, 100, 100)
AGENT_COLOR = (0, 255, 0)
KEY_COLOR = (255, 255, 0)
DOOR_COLOR = (255, 0, 0)

# Constants
CELL_SIZE = 60
TITLE_HEIGHT = 60
WINDOW = None

# Load sprites
def load_sprites():
    """Load sprites for visualization"""
    sprites = {}
    try:
        sprites_dir = project_root / "sprites"
        
        # Load agent sprite
        agent_path = sprites_dir / "keydoor" / "agent.png"
        if agent_path.exists():
            agent_surface = pygame.image.load(str(agent_path))
            agent_surface = pygame.transform.scale(agent_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['agent'] = agent_surface
        else:
            print(f"Warning: {agent_path} not found")
            
        # Load key sprite
        key_path = sprites_dir / "keydoor" / "keys.png"
        if key_path.exists():
            key_surface = pygame.image.load(str(key_path))
            key_surface = pygame.transform.scale(key_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['key'] = key_surface
        else:
            print(f"Warning: {key_path} not found")
            
        # Load door sprite
        door_path = sprites_dir / "keydoor" / "door.png"
        if door_path.exists():
            door_surface = pygame.image.load(str(door_path))
            door_surface = pygame.transform.scale(door_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['door'] = door_surface
        else:
            print(f"Warning: {door_path} not found")
            
    except pygame.error as e:
        print(f"Error loading sprites: {e}")
        
    return sprites

# Load sprites
SPRITES = load_sprites()

def resize_window(grid_size):
    """Resize pygame window based on grid size"""
    global WINDOW
    window_size = grid_size * CELL_SIZE
    WINDOW = pygame.display.set_mode((window_size, window_size + TITLE_HEIGHT))
    pygame.display.set_caption(f"SLM KeyDoor Agent - Template {grid_size}x{grid_size}")
    print(f"🔄 Window resized to {grid_size}x{grid_size} ({window_size}x{window_size + TITLE_HEIGHT})")

def draw_grid(env, template_name="Template", step_info=""):
    """Draw the environment grid"""
    grid_size = env.current_template['grid_size']
    
    # Resize window if needed
    if WINDOW is None:
        resize_window(grid_size)
    else:
        current_window_size = WINDOW.get_width()
        expected_window_size = grid_size * CELL_SIZE
        if current_window_size != expected_window_size:
            resize_window(grid_size)

    WINDOW.fill(WHITE)

    # Draw title at the top
    font = pygame.font.Font(None, 36)
    title_text = font.render(template_name, True, BLACK)
    window_width = grid_size * CELL_SIZE
    title_rect = title_text.get_rect(center=(window_width // 2, TITLE_HEIGHT // 2))
    WINDOW.blit(title_text, title_rect)
    
    # Draw step info below title
    if step_info:
        step_font = pygame.font.Font(None, 24)
        step_text = step_font.render(step_info, True, BLACK)
        step_rect = step_text.get_rect(center=(window_width // 2, TITLE_HEIGHT + 15))
        WINDOW.blit(step_text, step_rect)

    # Draw grid lines (offset by title height)
    for i in range(grid_size + 1):
        pygame.draw.line(WINDOW, BLACK, (i * CELL_SIZE, TITLE_HEIGHT), (i * CELL_SIZE, grid_size * CELL_SIZE + TITLE_HEIGHT), 2)
        pygame.draw.line(WINDOW, BLACK, (0, i * CELL_SIZE + TITLE_HEIGHT), (grid_size * CELL_SIZE, i * CELL_SIZE + TITLE_HEIGHT), 2)

    # Draw background (walls/floor only)
    grid_data = env.current_template['grid']
    for r in range(grid_size):
        for c in range(grid_size):
            if r < len(grid_data) and c < len(grid_data[r]):
                cell_type = grid_data[r][c]
                if cell_type == 1:  # Wall
                    wall_rect = pygame.Rect(c * CELL_SIZE + 2, r * CELL_SIZE + TITLE_HEIGHT + 2,
                                          CELL_SIZE - 4, CELL_SIZE - 4)
                    pygame.draw.rect(WINDOW, WALL_COLOR, wall_rect)
                    pygame.draw.rect(WINDOW, BLACK, wall_rect, 2)
                else:
                    floor_rect = pygame.Rect(c * CELL_SIZE + 2, r * CELL_SIZE + TITLE_HEIGHT + 2,
                                           CELL_SIZE - 4, CELL_SIZE - 4)
                    pygame.draw.rect(WINDOW, FLOOR_COLOR, floor_rect)
                    pygame.draw.rect(WINDOW, BLACK, floor_rect, 1)

    # Draw door (if not open)
    if not env.door_open:
        door_pos = env.current_template['door_pos']
        dr, dc = door_pos[0], door_pos[1]
        px = dc * CELL_SIZE + CELL_SIZE // 2
        py = dr * CELL_SIZE + CELL_SIZE // 2 + TITLE_HEIGHT
        if 'door' in SPRITES:
            sprite_rect = SPRITES['door'].get_rect(center=(px, py))
            WINDOW.blit(SPRITES['door'], sprite_rect)
        else:
            door_rect = pygame.Rect(px - CELL_SIZE // 3, py - CELL_SIZE // 3,
                                  CELL_SIZE // 1.5, CELL_SIZE // 1.5)
            pygame.draw.rect(WINDOW, DOOR_COLOR, door_rect)
            pygame.draw.rect(WINDOW, BLACK, door_rect, 2)

    # Draw remaining keys
    for key_pos in env.current_key_positions:
        # Convert observation coordinates to template coordinates
        start_row = (env.obs_grid_size - grid_size) // 2
        start_col = (env.obs_grid_size - grid_size) // 2
        template_r = key_pos[0] - start_row
        template_c = key_pos[1] - start_col
        
        if 0 <= template_r < grid_size and 0 <= template_c < grid_size:
            px = template_c * CELL_SIZE + CELL_SIZE // 2
            py = template_r * CELL_SIZE + CELL_SIZE // 2 + TITLE_HEIGHT
            if 'key' in SPRITES:
                sprite_rect = SPRITES['key'].get_rect(center=(px, py))
                WINDOW.blit(SPRITES['key'], sprite_rect)
            else:
                pygame.draw.circle(WINDOW, KEY_COLOR, (px, py), CELL_SIZE // 3)
                pygame.draw.circle(WINDOW, BLACK, (px, py), CELL_SIZE // 3, 2)

    # Draw agent
    # Convert observation coordinates to template coordinates
    start_row = (env.obs_grid_size - grid_size) // 2
    start_col = (env.obs_grid_size - grid_size) // 2
    agent_template_r = env.agent_pos[0] - start_row
    agent_template_c = env.agent_pos[1] - start_col
    
    if 0 <= agent_template_r < grid_size and 0 <= agent_template_c < grid_size:
        px = agent_template_c * CELL_SIZE + CELL_SIZE // 2
        py = agent_template_r * CELL_SIZE + CELL_SIZE // 2 + TITLE_HEIGHT
        if 'agent' in SPRITES:
            sprite_rect = SPRITES['agent'].get_rect(center=(px, py))
            WINDOW.blit(SPRITES['agent'], sprite_rect)
        else:
            pygame.draw.circle(WINDOW, AGENT_COLOR, (px, py), CELL_SIZE // 3)
            pygame.draw.circle(WINDOW, BLACK, (px, py), CELL_SIZE // 3, 2)

    pygame.display.update()

def next_template(current_template, all_templates):
    """Move to next template"""
    return (current_template + 1) % len(all_templates)

def get_template_name(template_id, template_type):
    """Get formatted template name"""
    return f"SLM Template {template_id} ({template_type})"

def reset_episode(env, agent, template_id, episode_num):
    """Reset environment for new episode"""
    obs, info = env.reset(template_id=template_id)
    agent.reset_stats()
    episode_reward = 0
    episode_length = 0
    done = False
    return obs, episode_reward, episode_length, done

def test_slm_pygame():
    """Test SLM KeyDoor Agent with pygame visualization"""
    print("🎮 Testing SLM KeyDoor Agent with Pygame Visualization")
    print("=" * 60)
    
    # Create environment and agent
    env = KeyDoorSLMEnv(grid_size=8, max_steps=200)
    agent = SLMKeyDoorAgent(model_name="llama3.2:3b", temperature=0.1)
    print(f"Environment: KeyDoorSLMEnv")
    print(f"Agent: SLMKeyDoorAgent with Llama 3.2 3B")
    
    # Test templates
    training_templates = [1, 2, 3, 4, 5, 6]  # T1-T6 (training)
    novel_templates = [7, 8, 9, 10]          # T7-T10 (novel)
    all_templates = training_templates + novel_templates
    
    # Configuration
    auto_template_change = True
    max_steps_per_episode = 200
    stuck_threshold = 50  # If no progress for 50 steps, move to next template
    
    print(f"\n🎮 Pygame Visualization Controls:")
    print(f"  - AUTO PLAY ENABLED by default (agent moves automatically)")
    print(f"  - Press SPACE to step through one action manually")
    print(f"  - Press A to toggle auto play on/off")
    print(f"  - Press P to pause and examine current state")
    print(f"  - Press 1-0 to go directly to template 1-10")
    print(f"  - Press R to reset current template")
    print(f"  - Press ESC to exit")
    print(f"\n🔄 Auto Template Change:")
    print(f"  - Templates auto-change when agent succeeds, fails, or gets stuck")
    print(f"  - Max steps per episode: {max_steps_per_episode}")
    print(f"  - Stuck threshold: {stuck_threshold} steps without progress")
    
    clock = pygame.time.Clock()
    current_template = 0
    current_episode = 0
    running = True
    auto_play = True  # Default to auto play
    
    # Start with first template
    template_id = all_templates[current_template]
    obs, episode_reward, episode_length, done = reset_episode(env, agent, template_id, current_episode)
    
    # Determine template type
    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
    
    # Draw initial state
    template_name = get_template_name(template_id, template_type)
    draw_grid(env, template_name, f"Episode {current_episode} - Step 0")
    pygame.display.update()
    
    print(f"🎮 Starting with Template {template_id} ({template_type}), Episode {current_episode}")
    print(f"📊 Template Info: {'Training (T1-T6)' if template_id in training_templates else 'Novel (T7-T10)'}")
    print(f"🎯 Expected Performance: {'100% success' if template_id in training_templates else '80%+ success'}")
    print(f"Controls: SPACE=step, A=auto play, P=pause, N=next episode, T=next template, R=reset, ESC=exit")
    
    last_progress_step = 0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Step through one action
                    if not done and episode_length < max_steps_per_episode:
                        # Get action from SLM agent
                        action = agent.act(env)
                        
                        # Log detailed action info
                        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "NO_OP"]
                        print(f"  Step {episode_length + 1}: Action={action_names[action]}")
                        
                        # Take action
                        obs, reward, done, info = env.step(action)
                        episode_reward += reward
                        episode_length += 1
                        
                        # Track progress
                        if env.keys_collected > 0 or env.door_open:
                            last_progress_step = episode_length
                        
                        # Log step results
                        print(f"    Reward: {reward:.3f}, Total: {episode_reward:.3f}, Done: {done}")
                        print(f"    Keys: {env.keys_collected}/{env.total_keys}, Door: {'OPEN' if env.door_open else 'CLOSED'}")
                        
                        # Draw updated state
                        step_info = f"Episode {current_episode} - Step {episode_length} - Keys: {env.keys_collected}/{env.total_keys}"
                        draw_grid(env, template_name, step_info)
                        pygame.display.update()
                        
                        # Check if episode should end
                        should_end = (done or 
                                    episode_length >= max_steps_per_episode or 
                                    (episode_length - last_progress_step) >= stuck_threshold)
                        
                        if should_end:
                            if done:
                                success = "✅ SUCCESS" if env._are_all_tasks_completed() else "❌ FAILED"
                                reason = "COMPLETED"
                            elif episode_length >= max_steps_per_episode:
                                success = "⏰ TIMEOUT"
                                reason = f"MAX STEPS ({max_steps_per_episode})"
                            else:
                                success = "🔄 STUCK"
                                reason = f"NO PROGRESS ({stuck_threshold} steps)"
                            
                            print(f"\n📊 Episode Results:")
                            print(f"  - {success} ({reason})")
                            print(f"  - Steps: {episode_length}")
                            print(f"  - Keys Collected: {env.keys_collected}/{env.total_keys}")
                            print(f"  - Door Status: {'OPEN' if env.door_open else 'CLOSED'}")
                            print(f"  - Agent Position: {tuple(env.agent_pos)}")
                            print(f"  - Door Position: {tuple(env.current_door_position)}")
                            print(f"  - Agent Performance: {agent.get_performance_stats()}")
                            
                            # Add a delay to see the final state
                            if success == "✅ SUCCESS":
                                print(f"  - 🎉 SUCCESS! Waiting 3 seconds to show final state...")
                                pygame.time.wait(3000)  # Wait 3 seconds
                            
                            if auto_template_change:
                                # Move to next template
                                current_template = next_template(current_template, all_templates)
                                template_id = all_templates[current_template]
                                template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                                current_episode += 1
                                
                                obs, episode_reward, episode_length, done = reset_episode(env, agent, template_id, current_episode)
                                template_name = get_template_name(template_id, template_type)
                                draw_grid(env, template_name, f"Episode {current_episode} - Step 0")
                                pygame.display.update()
                                
                                print(f"\n🔄 Moving to Template {template_id} ({template_type}), Episode {current_episode}")
                                last_progress_step = 0
                            else:
                                done = True
                
                elif event.key == pygame.K_a:
                    # Toggle auto play
                    auto_play = not auto_play
                    print(f"🔄 Auto play {'ENABLED' if auto_play else 'DISABLED'}")
                
                elif event.key == pygame.K_p:
                    # Pause and show current state
                    print(f"\n⏸️  PAUSED - Current State:")
                    print(f"  - Agent Position: {tuple(env.agent_pos)}")
                    print(f"  - Keys Collected: {env.keys_collected}/{env.total_keys}")
                    print(f"  - Door Status: {'OPEN' if env.door_open else 'CLOSED'}")
                    print(f"  - Door Position: {tuple(env.current_door_position)}")
                    print(f"  - Current Task: {env.get_current_navigation_task()}")
                    print(f"  - Steps: {episode_length}")
                    print(f"  - Press any key to continue...")
                    # Wait for any key press
                    waiting = True
                    while waiting:
                        for pause_event in pygame.event.get():
                            if pause_event.type == pygame.KEYDOWN:
                                waiting = False
                                break
                        pygame.time.wait(100)
                
                elif event.key == pygame.K_r:
                    # Reset current template
                    current_episode += 1
                    obs, episode_reward, episode_length, done = reset_episode(env, agent, template_id, current_episode)
                    template_name = get_template_name(template_id, template_type)
                    draw_grid(env, template_name, f"Episode {current_episode} - Step 0")
                    pygame.display.update()
                    print(f"🔄 Reset Template {template_id}, Episode {current_episode}")
                    last_progress_step = 0
                
                elif event.key == pygame.K_t:
                    # Next template
                    current_template = next_template(current_template, all_templates)
                    template_id = all_templates[current_template]
                    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                    current_episode += 1
                    
                    obs, episode_reward, episode_length, done = reset_episode(env, agent, template_id, current_episode)
                    template_name = get_template_name(template_id, template_type)
                    draw_grid(env, template_name, f"Episode {current_episode} - Step 0")
                    pygame.display.update()
                    
                    print(f"🔄 Moved to Template {template_id} ({template_type}), Episode {current_episode}")
                    last_progress_step = 0
                
                elif event.key >= pygame.K_1 and event.key <= pygame.K_0:
                    # Go to specific template (1-0)
                    template_num = (event.key - pygame.K_1 + 1) % 10
                    if template_num == 0:
                        template_num = 10
                    
                    if template_num <= len(all_templates):
                        current_template = template_num - 1
                        template_id = all_templates[current_template]
                        template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                        current_episode += 1
                        
                        obs, episode_reward, episode_length, done = reset_episode(env, agent, template_id, current_episode)
                        template_name = get_template_name(template_id, template_type)
                        draw_grid(env, template_name, f"Episode {current_episode} - Step 0")
                        pygame.display.update()
                        
                        print(f"🔄 Moved to Template {template_id} ({template_type}), Episode {current_episode}")
                        last_progress_step = 0
        
        # Auto play
        if auto_play and not done and episode_length < max_steps_per_episode:
            # Get action from SLM agent
            action = agent.act(env)
            
            # Take action
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Track progress
            if env.keys_collected > 0 or env.door_open:
                last_progress_step = episode_length
            
            # Draw updated state
            step_info = f"Episode {current_episode} - Step {episode_length} - Keys: {env.keys_collected}/{env.total_keys}"
            draw_grid(env, template_name, step_info)
            pygame.display.update()
            
            # Check if episode should end
            should_end = (done or 
                        episode_length >= max_steps_per_episode or 
                        (episode_length - last_progress_step) >= stuck_threshold)
            
            if should_end:
                if done:
                    success = "✅ SUCCESS" if env._are_all_tasks_completed() else "❌ FAILED"
                    reason = "COMPLETED"
                elif episode_length >= max_steps_per_episode:
                    success = "⏰ TIMEOUT"
                    reason = f"MAX STEPS ({max_steps_per_episode})"
                else:
                    success = "🔄 STUCK"
                    reason = f"NO PROGRESS ({stuck_threshold} steps)"
                
                print(f"\n📊 Episode Results:")
                print(f"  - {success} ({reason})")
                print(f"  - Steps: {episode_length}")
                print(f"  - Keys Collected: {env.keys_collected}/{env.total_keys}")
                print(f"  - Door Status: {'OPEN' if env.door_open else 'CLOSED'}")
                print(f"  - Agent Position: {tuple(env.agent_pos)}")
                print(f"  - Door Position: {tuple(env.current_door_position)}")
                print(f"  - Agent Performance: {agent.get_performance_stats()}")
                
                # Add a delay to see the final state
                if success == "✅ SUCCESS":
                    print(f"  - 🎉 SUCCESS! Waiting 3 seconds to show final state...")
                    pygame.time.wait(3000)  # Wait 3 seconds
                
                if auto_template_change:
                    # Move to next template
                    current_template = next_template(current_template, all_templates)
                    template_id = all_templates[current_template]
                    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                    current_episode += 1
                    
                    obs, episode_reward, episode_length, done = reset_episode(env, agent, template_id, current_episode)
                    template_name = get_template_name(template_id, template_type)
                    draw_grid(env, template_name, f"Episode {current_episode} - Step 0")
                    pygame.display.update()
                    
                    print(f"\n🔄 Moving to Template {template_id} ({template_type}), Episode {current_episode}")
                    last_progress_step = 0
                else:
                    done = True
            
            # Control frame rate for auto play (faster for better experience)
            clock.tick(5)  # 5 FPS for auto play (increased from 2)
        
        clock.tick(60)  # 60 FPS for manual control
    
    pygame.quit()
    print("\n🎮 Pygame visualization ended")

if __name__ == "__main__":
    test_slm_pygame()
