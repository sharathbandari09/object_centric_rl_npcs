#!/usr/bin/env python3
"""
Pygame visualization for BC Abstract (entity-based) model on KeyDoor.
- Dynamic window resizes to actual grid size (no outer whitespace)
- Door drawn as fixed, unpickable object from env state
- Agent drawn from env.agent_pos
- Logs mirror BC Raw pygame test
"""

import sys
import numpy as np
import pygame
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from envs.keydoor.keydoor_env import KeyDoorEnv


class AttentionPool(nn.Module):
    """Attention pooling for entity features"""
    def __init__(self, entity_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Linear(entity_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, entities, mask):
        # entities: (batch_size, max_entities, entity_dim)
        # mask: (batch_size, max_entities)

        # Compute attention weights
        attn_weights = self.proj(torch.tanh(self.query(entities)))  # (batch_size, max_entities, 1)
        attn_weights = attn_weights.squeeze(-1)  # (batch_size, max_entities)

        # Apply mask
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Weighted sum
        pooled = torch.sum(entities * attn_weights.unsqueeze(-1), dim=1)  # (batch_size, entity_dim)
        return pooled


class BCAbstract(nn.Module):
    def __init__(self, entity_dim=10, max_entities=16, n_actions=6, hidden_dim=256):
        super().__init__()
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attn_pool = AttentionPool(hidden_dim, hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.max_entities = max_entities
        self.hidden_dim = hidden_dim

    def forward(self, entities, mask):
        batch_size, max_entities, entity_dim = entities.shape
        entities_flat = entities.view(-1, entity_dim)
        entity_features = self.entity_encoder(entities_flat)
        entity_features = entity_features.view(batch_size, max_entities, self.hidden_dim)
        pooled = self.attn_pool(entity_features, mask)
        return self.policy_head(pooled)


# Pygame globals
pygame.init()
CELL_SIZE = 50
TITLE_HEIGHT = 40
WINDOW = None

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)


def load_sprites():
    sprites = {}
    sprites_dir = project_root / "sprites"
    try:
        p = sprites_dir / "keydoor" / "agent.png"
        if p.exists():
            s = pygame.image.load(str(p))
            sprites['agent'] = pygame.transform.scale(s, (CELL_SIZE - 4, CELL_SIZE - 4))
        p = sprites_dir / "keydoor" / "keys.png"
        if p.exists():
            s = pygame.image.load(str(p))
            sprites['key'] = pygame.transform.scale(s, (CELL_SIZE - 4, CELL_SIZE - 4))
        p = sprites_dir / "keydoor" / "door.png"
        if p.exists():
            s = pygame.image.load(str(p))
            sprites['door'] = pygame.transform.scale(s, (CELL_SIZE - 4, CELL_SIZE - 4))
    except pygame.error as e:
        print(f"Error loading sprites: {e}")
    return sprites


SPRITES = load_sprites()


def resize_window(grid_size: int):
    global WINDOW
    window_size = grid_size * CELL_SIZE
    WINDOW = pygame.display.set_mode((window_size, window_size + TITLE_HEIGHT))
    pygame.display.set_caption(f"BC Abstract Model Test - KeyDoor Environment ({grid_size}x{grid_size})")


essential_action_names = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "NO_OP"]


def draw_grid(obs, template_name: str, env: KeyDoorEnv):
    global WINDOW
    grid_size = env.grid_size

    if WINDOW is None or WINDOW.get_width() != grid_size * CELL_SIZE:
        resize_window(grid_size)

    WINDOW.fill(WHITE)

    # Title
    font = pygame.font.Font(None, 36)
    title_text = font.render(template_name, True, BLACK)
    window_width = grid_size * CELL_SIZE
    title_rect = title_text.get_rect(center=(window_width // 2, TITLE_HEIGHT // 2))
    WINDOW.blit(title_text, title_rect)

    # Grid lines
    for i in range(grid_size + 1):
        pygame.draw.line(WINDOW, BLACK, (i * CELL_SIZE, TITLE_HEIGHT), (i * CELL_SIZE, grid_size * CELL_SIZE + TITLE_HEIGHT), 2)
        pygame.draw.line(WINDOW, BLACK, (0, i * CELL_SIZE + TITLE_HEIGHT), (grid_size * CELL_SIZE, i * CELL_SIZE + TITLE_HEIGHT), 2)

    # Background (walls/floor)
    grid = obs['grid']
    for r in range(grid_size):
        for c in range(grid_size):
            rect = pygame.Rect(c * CELL_SIZE + 2, r * CELL_SIZE + TITLE_HEIGHT + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            if grid[r, c] == 1:  # wall
                pygame.draw.rect(WINDOW, GRAY, rect)
                pygame.draw.rect(WINDOW, BLACK, rect, 2)
            else:
                pygame.draw.rect(WINDOW, WHITE, rect)
                pygame.draw.rect(WINDOW, BLACK, rect, 1)

    # Door (fixed, unpickable): draw from env state if not opened
    if hasattr(env, 'current_door_position') and not env.door_open:
        dr, dc = int(env.current_door_position[0]), int(env.current_door_position[1])
        px = dc * CELL_SIZE + CELL_SIZE // 2
        py = dr * CELL_SIZE + CELL_SIZE // 2 + TITLE_HEIGHT
        if 'door' in SPRITES:
            WINDOW.blit(SPRITES['door'], SPRITES['door'].get_rect(center=(px, py)))
        else:
            rect = pygame.Rect(px - CELL_SIZE // 3, py - CELL_SIZE // 3, CELL_SIZE // 1.5, CELL_SIZE // 1.5)
            pygame.draw.rect(WINDOW, RED, rect)
            pygame.draw.rect(WINDOW, BLACK, rect, 2)

    # Keys from grid state (only if present)
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] == 3:
                px = c * CELL_SIZE + CELL_SIZE // 2
                py = r * CELL_SIZE + CELL_SIZE // 2 + TITLE_HEIGHT
                if 'key' in SPRITES:
                    WINDOW.blit(SPRITES['key'], SPRITES['key'].get_rect(center=(px, py)))
                else:
                    pygame.draw.circle(WINDOW, YELLOW, (px, py), CELL_SIZE // 3)
                    pygame.draw.circle(WINDOW, BLACK, (px, py), CELL_SIZE // 3, 2)

    # Agent from env.agent_pos (so door never appears as picked up)
    ar, ac = int(env.agent_pos[0]), int(env.agent_pos[1])
    px = ac * CELL_SIZE + CELL_SIZE // 2
    py = ar * CELL_SIZE + CELL_SIZE // 2 + TITLE_HEIGHT
    if 'agent' in SPRITES:
        WINDOW.blit(SPRITES['agent'], SPRITES['agent'].get_rect(center=(px, py)))
    else:
        pygame.draw.circle(WINDOW, BLUE, (px, py), CELL_SIZE // 3)
        pygame.draw.circle(WINDOW, BLACK, (px, py), CELL_SIZE // 3, 2)

    pygame.display.update()


def act_from_entities(model: BCAbstract, obs, device):
    entities = obs['entities']
    entity_mask = obs['entity_mask']
    ent_tensor = torch.tensor(entities, dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.tensor(entity_mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(ent_tensor, mask_tensor)
        action = torch.argmax(logits, dim=-1).item()
    return action, logits


def test_bc_abstract_pygame():
    print("🎮 Testing BC Abstract Model with Pygame Visualization")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create environment
    env = KeyDoorEnv(max_steps=50)
    print(f"Environment: KeyDoor with variable grid sizes")

    # Load model
    model_path = project_root / "experiments" / "exp_bc_abstract_keydoor_seed0" / "bc_abstract_best.pth"
    model = BCAbstract().to(device)
    model.eval()
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first using: python scripts/train_bc_abstract_fixed_v2.py")
        return
    print(f"Loading model from: {model_path}")
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("✅ Model loaded successfully")

    training_templates = [1, 2, 3, 4, 5, 6]
    novel_templates = [7, 8, 9, 10]
    all_templates = training_templates + novel_templates

    auto_template_change = True
    max_steps_per_episode = 50
    stuck_threshold = 10_000

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
    current_idx = 0
    current_episode = 0
    running = True
    auto_play = True

    # Start with first template
    template_id = all_templates[current_idx]
    obs = env.reset(template_id=template_id, seed=current_episode)
    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
    template_name = f"Template {template_id} ({template_type})"

    # Draw initial state
    draw_grid(obs, template_name, env)
    pygame.display.update()

    episode_reward = 0.0
    episode_length = 0
    done = False
    last_reward = 0.0
    steps_since_progress = 0

    print(f"🎮 Starting with Template {template_id} ({template_type}), Episode {current_episode}")
    print(f"📊 Template Info: {'Training (T1-T6)' if template_id in training_templates else 'Novel (T7-T10)'}")
    print(f"🎯 Expected Performance: {'~100% success' if template_id in training_templates else 'Low/0% success'}")
    print(f"Controls: SPACE=step, A=auto play, N=next episode, T=next template, R=reset, ESC=exit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if not done and episode_length < 100:
                        action, logits = act_from_entities(model, obs, device)
                        print(f"  Step {episode_length + 1}: Action={essential_action_names[action]} (logits: {logits[0].cpu().numpy()})")
                        obs, reward, done, info = env.step(action)
                        episode_reward += reward
                        episode_length += 1
                        steps_since_progress = 0 if reward > last_reward else steps_since_progress + 1
                        last_reward = max(last_reward, reward)
                        draw_grid(obs, template_name, env)
                        pygame.display.update()
                        should_end = (done or episode_length >= max_steps_per_episode or steps_since_progress >= stuck_threshold)
                        if should_end:
                            if done:
                                success = "✅ SUCCESS" if info.get('success', False) else "❌ FAILED"
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
                                current_idx = (current_idx + 1) % len(all_templates)
                                current_episode = 0
                                template_id = all_templates[current_idx]
                                template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                                print(f"🔄 Auto-changing to Template {template_id} ({template_type})")
                                obs = env.reset(template_id=template_id, seed=current_episode)
                                template_name = f"Template {template_id} ({template_type})"
                                episode_reward = 0.0
                                episode_length = 0
                                done = False
                                last_reward = 0.0
                                steps_since_progress = 0
                                draw_grid(obs, template_name, env)
                                pygame.display.update()
                            else:
                                done = True
                elif event.key == pygame.K_a:
                    auto_play = not auto_play
                    print(f"Auto play: {'ON' if auto_play else 'OFF'}")
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0]:
                    template_num = int(pygame.key.name(event.key))
                    if template_num in all_templates:
                        current_idx = all_templates.index(template_num)
                        current_episode = 0
                        template_id = template_num
                        obs = env.reset(template_id=template_id, seed=current_episode)
                        episode_reward = 0.0
                        episode_length = 0
                        done = False
                        last_reward = 0.0
                        steps_since_progress = 0
                        template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                        print(f"🔄 Template {template_id} ({template_type})")
                        print(f"📊 Template Info: {'Training (T1-T6)' if template_id in training_templates else 'Novel (T7-T10)'}")
                        print(f"🎯 Expected Performance: {'~100% success' if template_id in training_templates else 'Low/0% success'}")
                        template_name = f"Template {template_id} ({template_type})"
                        draw_grid(obs, template_name, env)
                        pygame.display.update()
                elif event.key == pygame.K_r:
                    current_episode = 0
                    template_id = all_templates[current_idx]
                    obs = env.reset(template_id=template_id, seed=current_episode)
                    episode_reward = 0.0
                    episode_length = 0
                    done = False
                    last_reward = 0.0
                    steps_since_progress = 0
                    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                    print(f"🔄 Reset Template {template_id} ({template_type})")
                    template_name = f"Template {template_id} ({template_type})"
                    draw_grid(obs, template_name, env)
                    pygame.display.update()

        # Auto play mode
        if auto_play and not done and episode_length < max_steps_per_episode and steps_since_progress < stuck_threshold:
            action, logits = act_from_entities(model, obs, device)
            if episode_length % 5 == 0:
                print(f"  Auto Step {episode_length + 1}: Action={essential_action_names[action]}")
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            steps_since_progress = 0 if reward > last_reward else steps_since_progress + 1
            last_reward = max(last_reward, reward)
            draw_grid(obs, template_name, env)
            pygame.display.update()

            should_end = (done or episode_length >= max_steps_per_episode or steps_since_progress >= stuck_threshold)
            if should_end:
                if done:
                    success = "✅ SUCCESS" if info.get('success', False) else "❌ FAILED"
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
                    current_idx = (current_idx + 1) % len(all_templates)
                    current_episode = 0
                    template_id = all_templates[current_idx]
                    template_type = "TRAINING" if template_id in training_templates else "NOVEL"
                    print(f"🔄 Auto-changing to Template {template_id} ({template_type})")
                    obs = env.reset(template_id=template_id, seed=current_episode)
                    template_name = f"Template {template_id} ({template_type})"
                    episode_reward = 0.0
                    episode_length = 0
                    done = False
                    last_reward = 0.0
                    steps_since_progress = 0
                    draw_grid(obs, template_name, env)
                    pygame.display.update()
                else:
                    auto_play = False

            pygame.time.delay(200)

        clock.tick(60)

    # Clean exit
    try:
        env.close()
    finally:
        pygame.display.quit()
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    test_bc_abstract_pygame()

