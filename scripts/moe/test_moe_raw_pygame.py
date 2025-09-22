#!/usr/bin/env python3
"""
Test MoE Raw model with pygame visualization (mirrors BC Raw visualization)
"""

import pygame
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from envs.keydoor.keydoor_env import KeyDoorEnv
from agents.moe.moe_raw import MoERaw

# Pygame setup
pygame.init()
CELL_SIZE = 50
TITLE_HEIGHT = 40  # Space for title at top
# Window size will be set dynamically based on grid size
WINDOW = None

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
PURPLE = (128, 0, 128)

# Entity colors
AGENT_COLOR = BLUE
KEY_COLOR = YELLOW
DOOR_COLOR = RED
WALL_COLOR = GRAY
FLOOR_COLOR = WHITE

# Load sprite images
def load_sprites():
    """Load sprite images from sprites folder"""
    sprites = {}
    sprites_dir = project_root / "sprites"
    
    try:
        # Load agent sprite
        agent_path = sprites_dir / "keydoor" / "agent.png"
        if agent_path.exists():
            agent_surface = pygame.image.load(str(agent_path))
            # Scale to fit cell size
            agent_surface = pygame.transform.scale(agent_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['agent'] = agent_surface
        
        # Load key sprite
        key_path = sprites_dir / "keydoor" / "keys.png"
        if key_path.exists():
            key_surface = pygame.image.load(str(key_path))
            # Scale to fit cell size
            key_surface = pygame.transform.scale(key_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['key'] = key_surface
        
        # Load door sprite
        door_path = sprites_dir / "keydoor" / "door.png"
        if door_path.exists():
            door_surface = pygame.image.load(str(door_path))
            # Scale to fit cell size
            door_surface = pygame.transform.scale(door_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['door'] = door_surface
    except pygame.error:
        pass
        
    return sprites

# Load sprites
SPRITES = load_sprites()

def resize_window(grid_size):
    """Resize pygame window based on grid size"""
    global WINDOW
    window_size = grid_size * CELL_SIZE
    # Create a new window with the correct size
    WINDOW = pygame.display.set_mode((window_size, window_size + TITLE_HEIGHT))
    pygame.display.set_caption(f"MoE Raw Model Test - KeyDoor Environment ({grid_size}x{grid_size})")
    print(f"🔄 Window resized to {grid_size}x{grid_size} ({window_size}x{window_size + TITLE_HEIGHT})")

def draw_grid(obs, template_name="Template", grid_size=None, env=None):
    """Draw the environment grid from observation"""
    # Get grid size from environment if available, otherwise from observation
    if grid_size is None:
        if env is not None and hasattr(env, 'grid_size'):
            grid_size = env.grid_size  # Use actual template grid size
        else:
            grid = obs['grid']
            grid_size = grid.shape[0]  # Fallback

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

    # Draw grid lines (offset by title height)
    for i in range(grid_size + 1):
        pygame.draw.line(WINDOW, BLACK, (i * CELL_SIZE, TITLE_HEIGHT), (i * CELL_SIZE, grid_size * CELL_SIZE + TITLE_HEIGHT), 2)
        pygame.draw.line(WINDOW, BLACK, (0, i * CELL_SIZE + TITLE_HEIGHT), (grid_size * CELL_SIZE, i * CELL_SIZE + TITLE_HEIGHT), 2)

    # Draw background (walls/floor only)
    grid = obs['grid']
    for r in range(grid_size):
        for c in range(grid_size):
            cell_type = grid[r, c]
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

    # Draw fixed door from env state
    if env is not None:
        door_pos = tuple(env.current_door_position.tolist()) if hasattr(env, 'current_door_position') else None
        if door_pos is not None and not env.door_open:
            dr, dc = door_pos
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

    # Draw keys from grid state
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] == 3:  # Key
                px = c * CELL_SIZE + CELL_SIZE // 2
                py = r * CELL_SIZE + CELL_SIZE // 2 + TITLE_HEIGHT
                if 'key' in SPRITES:
                    sprite_rect = SPRITES['key'].get_rect(center=(px, py))
                    WINDOW.blit(SPRITES['key'], sprite_rect)
                else:
                    pygame.draw.circle(WINDOW, KEY_COLOR, (px, py), CELL_SIZE // 3)
                    pygame.draw.circle(WINDOW, BLACK, (px, py), CELL_SIZE // 3, 2)

    # Draw agent from env.agent_pos
    if env is not None and hasattr(env, 'agent_pos'):
        ar, ac = int(env.agent_pos[0]), int(env.agent_pos[1])
        px = ac * CELL_SIZE + CELL_SIZE // 2
        py = ar * CELL_SIZE + CELL_SIZE // 2 + TITLE_HEIGHT
        if 'agent' in SPRITES:
            sprite_rect = SPRITES['agent'].get_rect(center=(px, py))
            WINDOW.blit(SPRITES['agent'], sprite_rect)
        else:
            pygame.draw.circle(WINDOW, AGENT_COLOR, (px, py), CELL_SIZE // 3)
            pygame.draw.circle(WINDOW, BLACK, (px, py), CELL_SIZE // 3, 2)

    pygame.display.update()


def next_template(current_template, all_templates):
    return (current_template + 1) % len(all_templates)


def get_template_name(template_id, template_type):
    return f"Template {template_id} ({template_type})"


def reset_episode(env, template_id, episode_num):
    obs = env.reset(template_id=template_id, seed=episode_num)
    episode_reward = 0
    episode_length = 0
    done = False
    last_reward = 0
    steps_since_progress = 0
    return obs, episode_reward, episode_length, done, last_reward, steps_since_progress


def convert_obs_to_pixels(obs, max_size=12):
    grid = obs['grid']
    grid_size = grid.shape[0]
    pixel_obs = np.zeros((1, max_size, max_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            cell_type = grid[r, c]
            if cell_type == 1:
                pixel_obs[0, r, c] = 1.0
            elif cell_type == 2:
                pixel_obs[0, r, c] = 0.5
            elif cell_type == 3:
                pixel_obs[0, r, c] = 0.8
            elif cell_type == 4:
                pixel_obs[0, r, c] = 0.3
    return pixel_obs.reshape(1, 1, max_size, max_size)


def test_moe_raw_pygame():
    print("🎮 Testing MoE Raw Model with Pygame Visualization")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    env = KeyDoorEnv(max_steps=50)
    print(f"Environment: KeyDoor with variable grid sizes")

    # Create MoE Raw model
    model = MoERaw(n_experts=4, input_dim=64, output_dim=6, device=str(device))
    model.to(device)
    model.eval()

    # Load trained model
    model_path = project_root / "experiments" / "exp_moe_raw_keydoor_seed0" / "moe_raw_final.pth"
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first using: python scripts/train_moe_raw.py")
        return
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model loaded successfully")

    # Templates
    training_templates = [1, 2, 3, 4, 5, 6]
    novel_templates = [7, 8, 9, 10]
    all_templates = training_templates + novel_templates

    # Configuration
    auto_template_change = True
    max_steps_per_episode = 50
    stuck_threshold = 10_000  # If no progress for 10,000 steps, move to next template

    print(f"\n🎮 Pygame Visualization Controls:")
    print(f"  - AUTO PLAY ENABLED by default (agent moves automatically)")
    print(f"  - Press SPACE to step through one action manually")
    print(f"  - Press A to toggle auto play on/off")
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
    auto_play = True

    template_id = all_templates[current_template]
    obs = env.reset(template_id=template_id, seed=current_episode)
    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
    template_name = get_template_name(template_id, template_type)
    draw_grid(obs, template_name, env=env)
    pygame.display.update()

    episode_reward = 0
    episode_length = 0
    done = False
    last_reward = 0
    steps_since_progress = 0

    print(f"🎮 Starting with Template {template_id} ({template_type}), Episode {current_episode}")
    print(f"📊 Template Info: {'Training (T1-T6)' if template_id in training_templates else 'Novel (T7-T10)'}")
    print(f"🎯 Expected Performance: {'100% success' if template_id in training_templates else '0% success'}")
    print(f"Controls: SPACE=step, A=auto play, N=next episode, T=next template, R=reset, ESC=exit")

    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "NO_OP"]

    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if not done and episode_length < max_steps_per_episode:
                        pixel_obs = convert_obs_to_pixels(obs)
                        pixel_tensor = torch.tensor(pixel_obs, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            logits, _ = model(pixel_tensor)
                            action = torch.argmax(logits, dim=-1).item()
                        print(f"  Step {episode_length + 1}: Action={action_names[action]} (logits: {logits[0].cpu().numpy()})")
                        obs, reward, done, info = env.step(action)
                        episode_reward += reward
                        episode_length += 1
                        if reward > last_reward:
                            steps_since_progress = 0
                            last_reward = reward
                        else:
                            steps_since_progress += 1
                        draw_grid(obs, template_name, env=env)
                        pygame.display.update()
                        should_end = (done or episode_length >= max_steps_per_episode or steps_since_progress >= stuck_threshold)
                        if should_end:
                            if done:
                                success = "✅ SUCCESS" if episode_reward > 0.5 else "❌ FAILED"
                                reason = "COMPLETED"
                            elif episode_length >= max_steps_per_episode:
                                success = "⏰ TIMEOUT"
                                reason = f"MAX STEPS ({max_steps_per_episode})"
                            else:
                                success = "🔄 STUCK"
                                reason = f"NO PROGRESS ({stuck_threshold} steps)"
                            print(f"🎯 Episode finished: {success} | Reward: {episode_reward:.3f} | Length: {episode_length} | Reason: {reason}")
                            if 'success' in info:
                                print(f"    Final Success: {info['success']}")
                            if auto_template_change:
                                current_template = next_template(current_template, all_templates)
                                current_episode = 0
                                template_id = all_templates[current_template]
                                template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                                print(f"🔄 Auto-changing to Template {template_id} ({template_type})")
                                obs, episode_reward, episode_length, done, last_reward, steps_since_progress = reset_episode(env, template_id, current_episode)
                                template_name = get_template_name(template_id, template_type)
                                draw_grid(obs, template_name, env=env)
                                pygame.display.update()
                            else:
                                done = True
                elif event.key == pygame.K_a:
                    auto_play = not auto_play
                    print(f"Auto play: {'ON' if auto_play else 'OFF'}")
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0]:
                    template_num = int(pygame.key.name(event.key))
                    if template_num in all_templates:
                        current_template = all_templates.index(template_num)
                        current_episode = 0
                        template_id = template_num
                        obs = env.reset(template_id=template_id, seed=current_episode)
                        episode_reward = 0
                        episode_length = 0
                        done = False
                        template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                        print(f"🔄 Template {template_id} ({template_type})")
                        print(f"📊 Template Info: {'Training (T1-T6)' if template_id in training_templates else 'Novel (T7-T10)'}")
                        print(f"🎯 Expected Performance: {'100% success' if template_id in training_templates else '0% success'}")
                        template_name = get_template_name(template_id, template_type)
                        draw_grid(obs, template_name, env=env)
                        pygame.display.update()
                elif event.key == pygame.K_r:
                    current_episode = 0
                    template_id = all_templates[current_template]
                    obs = env.reset(template_id=template_id, seed=current_episode)
                    episode_reward = 0
                    episode_length = 0
                    done = False
                    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                    print(f"🔄 Reset Template {template_id} ({template_type})")
                    template_name = get_template_name(template_id, template_type)
                    draw_grid(obs, template_name, env=env)
                    pygame.display.update()

        # Auto play
        if auto_play and not done and episode_length < max_steps_per_episode and steps_since_progress < stuck_threshold:
            pixel_obs = convert_obs_to_pixels(obs)
            pixel_tensor = torch.tensor(pixel_obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits, _ = model(pixel_tensor)
                action = torch.argmax(logits, dim=-1).item()
            if episode_length % 5 == 0:
                print(f"  Auto Step {episode_length + 1}: Action={action_names[action]}")
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            if reward > last_reward:
                steps_since_progress = 0
                last_reward = reward
            else:
                steps_since_progress += 1
            draw_grid(obs, template_name, env=env)
            pygame.display.update()
            should_end = (done or episode_length >= max_steps_per_episode or steps_since_progress >= stuck_threshold)
            if should_end:
                if done:
                    success = "✅ SUCCESS" if episode_reward > 0.5 else "❌ FAILED"
                    reason = "COMPLETED"
                elif episode_length >= max_steps_per_episode:
                    success = "⏰ TIMEOUT"
                    reason = f"MAX STEPS ({max_steps_per_episode})"
                else:
                    success = "🔄 STUCK"
                    reason = f"NO PROGRESS ({stuck_threshold} steps)"
                print(f"🎯 Auto Episode finished: {success} | Reward: {episode_reward:.3f} | Length: {episode_length} | Reason: {reason}")
                if 'success' in info:
                    print(f"    Final Success: {info['success']}")
                if auto_template_change:
                    current_template = next_template(current_template, all_templates)
                    current_episode = 0
                    template_id = all_templates[current_template]
                    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                    print(f"🔄 Auto-changing to Template {template_id} ({template_type})")
                    obs, episode_reward, episode_length, done, last_reward, steps_since_progress = reset_episode(env, template_id, current_episode)
                    template_name = get_template_name(template_id, template_type)
                    draw_grid(obs, template_name, env=env)
                    pygame.display.update()
                else:
                    auto_play = False
            pygame.time.delay(200)

        pygame.time.Clock().tick(60)

    pygame.quit()
    env.close()
    print("\n🎮 Pygame visualization completed!")


if __name__ == "__main__":
    test_moe_raw_pygame()
