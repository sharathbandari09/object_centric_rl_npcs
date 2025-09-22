#!/usr/bin/env python3
"""
Test PPO Abstract model with Pygame visualization (mirrors PPO Raw Pygame)
"""

import pygame
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from envs.keydoor.keydoor_rl_env import KeyDoorRLEnv
from agents.ppo.ppo_abstract import PPOAbstract

# Pygame setup
pygame.init()
CELL_SIZE = 50
TITLE_HEIGHT = 40
WINDOW = None

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

AGENT_COLOR = BLUE
KEY_COLOR = YELLOW
DOOR_COLOR = RED
WALL_COLOR = GRAY
FLOOR_COLOR = WHITE

def load_sprites():
    sprites = {}
    sprites_dir = project_root / "sprites" / "keydoor"
    try:
        agent_path = sprites_dir / "agent.png"
        if agent_path.exists():
            agent_surface = pygame.image.load(str(agent_path))
            agent_surface = pygame.transform.scale(agent_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['agent'] = agent_surface
        key_path = sprites_dir / "keys.png"
        if key_path.exists():
            key_surface = pygame.image.load(str(key_path))
            key_surface = pygame.transform.scale(key_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['key'] = key_surface
        door_path = sprites_dir / "door.png"
        if door_path.exists():
            door_surface = pygame.image.load(str(door_path))
            door_surface = pygame.transform.scale(door_surface, (CELL_SIZE - 4, CELL_SIZE - 4))
            sprites['door'] = door_surface
    except pygame.error:
        pass
    return sprites

SPRITES = load_sprites()

def resize_window(grid_size):
    global WINDOW
    window_size = grid_size * CELL_SIZE
    WINDOW = pygame.display.set_mode((window_size, window_size + TITLE_HEIGHT))
    pygame.display.set_caption(f"PPO Abstract Model Test - KeyDoor RL Environment ({grid_size}x{grid_size})")
    print(f"Window resized to {grid_size}x{grid_size} ({window_size}x{window_size + TITLE_HEIGHT})")

def draw_grid(obs, template_name="Template", grid_size=None, env=None):
    if grid_size is None:
        if env is not None and hasattr(env, 'grid_size'):
            grid_size = env.grid_size
        else:
            grid = obs['grid']
            grid_size = grid.shape[0]

    if WINDOW is None:
        resize_window(grid_size)
    else:
        current_window_size = WINDOW.get_width()
        expected_window_size = grid_size * CELL_SIZE
        if current_window_size != expected_window_size:
            resize_window(grid_size)

    WINDOW.fill(WHITE)

    font = pygame.font.Font(None, 36)
    title_text = font.render(template_name, True, BLACK)
    window_width = grid_size * CELL_SIZE
    title_rect = title_text.get_rect(center=(window_width // 2, TITLE_HEIGHT // 2))
    WINDOW.blit(title_text, title_rect)

    for i in range(grid_size + 1):
        pygame.draw.line(WINDOW, BLACK, (i * CELL_SIZE, TITLE_HEIGHT), (i * CELL_SIZE, grid_size * CELL_SIZE + TITLE_HEIGHT), 2)
        pygame.draw.line(WINDOW, BLACK, (0, i * CELL_SIZE + TITLE_HEIGHT), (grid_size * CELL_SIZE, i * CELL_SIZE + TITLE_HEIGHT), 2)

    grid = obs['grid']
    for r in range(grid_size):
        for c in range(grid_size):
            cell_type = grid[r, c]
            if cell_type == 1:  # Wall
                wall_rect = pygame.Rect(c * CELL_SIZE + 2, r * CELL_SIZE + TITLE_HEIGHT + 2, CELL_SIZE - 4, CELL_SIZE - 4)
                pygame.draw.rect(WINDOW, WALL_COLOR, wall_rect)
                pygame.draw.rect(WINDOW, BLACK, wall_rect, 2)
            else:
                floor_rect = pygame.Rect(c * CELL_SIZE + 2, r * CELL_SIZE + TITLE_HEIGHT + 2, CELL_SIZE - 4, CELL_SIZE - 4)
                pygame.draw.rect(WINDOW, FLOOR_COLOR, floor_rect)
                pygame.draw.rect(WINDOW, BLACK, floor_rect, 1)

    # Draw door from env state if available (never moves)
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
                door_rect = pygame.Rect(px - CELL_SIZE // 3, py - CELL_SIZE // 3, CELL_SIZE // 1.5, CELL_SIZE // 1.5)
                pygame.draw.rect(WINDOW, DOOR_COLOR, door_rect)
                pygame.draw.rect(WINDOW, BLACK, door_rect, 2)

    # Draw keys from grid state
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] == 3:  # Key (only if still present)
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

def get_entity_tensors(obs, device):
    ent = torch.tensor(obs['entities'], dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(obs['entity_mask'], dtype=torch.bool).unsqueeze(0).to(device)
    return ent, mask

def test_ppo_abstract_pygame():
    print("Testing PPO Abstract Model with Pygame Visualization (RL-Optimized Environment)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = KeyDoorRLEnv(max_steps=10000)  # Very high limit to effectively remove it
    print("Environment: KeyDoor RL (RL-optimized with dense rewards)")

    model = PPOAbstract(entity_dim=10, max_entities=16, n_actions=6, hidden_dim=256)
    model.to(device)
    model.eval()

    model_path = project_root / "experiments" / "exp_ppo_abstract_rl_keydoor_seed0" / "ppo_abstract_final.pth"
    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        print("Please train the model first using: python scripts/train_ppo_abstract.py")
        return

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    if 'step' in checkpoint:
        print(f"Training step: {checkpoint['step']}")
    if 'metrics' in checkpoint:
        print(f"Final metrics: {checkpoint['metrics']}")

    training_templates = [1, 2, 3, 4, 5, 6]
    novel_templates = [7, 8, 9, 10]
    all_templates = training_templates + novel_templates

    auto_template_change = False
    max_steps_per_episode = 10000  # Very high limit to effectively remove it
    stuck_threshold = 1000  # Higher threshold since no max steps limit

    print("\nPygame Visualization Controls:")
    print("  - AUTO PLAY ENABLED by default (agent moves automatically)")
    print("  - Press SPACE to step through one action manually")
    print("  - Press A to toggle auto play on/off")
    print("  - Press N to start next episode (only when current episode is finished)")
    print("  - Press T to move to next template")
    print("  - Press R to reset current template/episode")
    print("  - Press 1-0 to go directly to template 1-10")
    print("  - Press ESC to exit")

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
    print(f"Starting with Template {template_id} ({template_type}), Episode {current_episode}")
    print("Controls: SPACE=step, A=auto play, N=next episode, T=next template, R=reset, ESC=exit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if not done:
                        ent, mask = get_entity_tensors(obs, device)
                        with torch.no_grad():
                            action, log_prob, value = model.get_action(ent, mask, deterministic=False)
                        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "NO_OP"]
                        print(f"  Step {episode_length + 1}: Pos={env.agent_pos} | Action={action_names[action]}")
                        obs, reward, done, info = env.step(action)
                        episode_reward += reward
                        episode_length += 1
                        if reward > last_reward:
                            steps_since_progress = 0
                            last_reward = reward
                        else:
                            steps_since_progress += 1
                        print(f"    Reward: {reward:.3f}, Total: {episode_reward:.3f}, Done: {done}")
                        if 'success' in info:
                            print(f"    Success: {info['success']}")
                        draw_grid(obs, template_name, env=env)
                        pygame.display.update()
                        should_end = (done or steps_since_progress >= stuck_threshold)
                        if should_end:
                            if done:
                                success = "SUCCESS" if env.door_open else "FAILED"
                                reason = "COMPLETED"
                            else:
                                success = "STUCK"
                                reason = f"NO PROGRESS ({stuck_threshold} steps)"
                            print(f"\nEpisode {current_episode} finished: {success} - {reason}")
                            print(f"Final Stats: Reward={episode_reward:.3f}, Steps={episode_length}")
                            print("Press N for next episode, T for next template, or R to reset")
                elif event.key == pygame.K_a:
                    auto_play = not auto_play
                    print(f"Auto play: {'ON' if auto_play else 'OFF'}")
                elif event.key == pygame.K_r:
                    current_episode = 0
                    obs, episode_reward, episode_length, done, last_reward, steps_since_progress = reset_episode(env, template_id, current_episode)
                    draw_grid(obs, template_name, env=env)
                    pygame.display.update()
                    print(f"Reset Template {template_id}, Episode {current_episode}")
                elif event.key == pygame.K_n:
                    if done or episode_length >= max_steps_per_episode or steps_since_progress >= stuck_threshold:
                        current_episode += 1
                        obs, episode_reward, episode_length, done, last_reward, steps_since_progress = reset_episode(env, template_id, current_episode)
                        draw_grid(obs, template_name, env=env)
                        pygame.display.update()
                        print(f"Next episode on Template {template_id}, Episode {current_episode}")
                    else:
                        print("Episode still in progress - wait for completion or press R to reset")
                elif event.key == pygame.K_t:
                    current_template = next_template(current_template, all_templates)
                    template_id = all_templates[current_template]
                    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                    template_name = get_template_name(template_id, template_type)
                    current_episode = 0
                    obs, episode_reward, episode_length, done, last_reward, steps_since_progress = reset_episode(env, template_id, current_episode)
                    draw_grid(obs, template_name, env=env)
                    pygame.display.update()
                    print(f"Moved to Template {template_id} ({template_type}), Episode {current_episode}")
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    template_idx = event.key - pygame.K_1
                    if template_idx < len(all_templates):
                        current_template = template_idx
                        template_id = all_templates[current_template]
                        template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                        template_name = get_template_name(template_id, template_type)
                        current_episode = 0
                        obs, episode_reward, episode_length, done, last_reward, steps_since_progress = reset_episode(env, template_id, current_episode)
                        draw_grid(obs, template_name, env=env)
                        pygame.display.update()
                        print(f"Jumped to Template {template_id} ({template_type}), Episode {current_episode}")
                elif event.key == pygame.K_0:
                    if len(all_templates) >= 10:
                        current_template = 9
                        template_id = all_templates[current_template]
                        template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                        template_name = get_template_name(template_id, template_type)
                        current_episode = 0
                        obs, episode_reward, episode_length, done, last_reward, steps_since_progress = reset_episode(env, template_id, current_episode)
                        draw_grid(obs, template_name, env=env)
                        pygame.display.update()
                        print(f"Jumped to Template {template_id} ({template_type}), Episode {current_episode}")

        if auto_play and not done and steps_since_progress < stuck_threshold:
            ent, mask = get_entity_tensors(obs, device)
            with torch.no_grad():
                action, log_prob, value = model.get_action(ent, mask, deterministic=False)
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
            should_end = (done or steps_since_progress >= stuck_threshold)
            if should_end:
                if done:
                    success = "SUCCESS" if env.door_open else "FAILED"
                    reason = "COMPLETED"
                else:
                    success = "STUCK"
                    reason = f"NO PROGRESS ({stuck_threshold} steps)"
                print(f"\nEpisode {current_episode} finished: {success} - {reason}")
                print(f"Final Stats: Reward={episode_reward:.3f}, Steps={episode_length}")
                print("Press N for next episode, T for next template, or R to reset")

        pygame.time.Clock().tick(5)

    print("\nPPO Abstract Pygame Test Complete")
    env.close()
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    test_ppo_abstract_pygame()
