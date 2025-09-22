#!/usr/bin/env python3
"""
Test PPO Abstract model using entity observations from KeyDoorRLEnv
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from envs.keydoor.keydoor_rl_env import KeyDoorRLEnv
from agents.ppo.ppo_abstract import PPOAbstract


def test_ppo_abstract():
    print("=== Testing PPO Abstract Model (RL-Optimized Environment) ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = PPOAbstract(entity_dim=10, max_entities=16, n_actions=6, hidden_dim=256).to(device)
    checkpoint = torch.load('experiments/exp_ppo_abstract_rl_keydoor_seed0/ppo_abstract_final.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from step {checkpoint['step_count']}")
    print(f"Final metrics: {checkpoint['metrics']}")

    # Create RL-optimized environment
    env = KeyDoorRLEnv(max_steps=200)

    # Test on training layouts (T1-T6)
    print("\n--- Training Layouts (T1-T6) ---")
    training_success = 0
    training_total = 0

    for template_id in range(1, 7):
        print(f"Testing template {template_id}...")
        template_success = 0

        for i in range(10):
            obs = env.reset(template_id=template_id, seed=i)
            done = False
            episode_reward = 0

            for t in range(200):
                entities_t = torch.tensor(obs['entities'], dtype=torch.float32).unsqueeze(0).to(device)
                mask_t = torch.tensor(obs['entity_mask'], dtype=torch.bool).unsqueeze(0).to(device)

                with torch.no_grad():
                    action, _, _ = model.get_action(entities_t, mask_t, deterministic=True)

                obs, reward, done, info = env.step(action)
                episode_reward += reward

                if done:
                    break

            if info.get('success', False):
                template_success += 1
                training_success += 1
            training_total += 1

        print(f"  Template {template_id}: {template_success}/10 successes")

    training_rate = training_success / training_total if training_total > 0 else 0.0
    print(f"Training Layouts Success Rate: {training_rate:.2f} ({training_success}/{training_total})")

    # Test on novel layouts (T7-T10)
    print("\n--- Novel Layouts (T7-T10) ---")
    novel_success = 0
    novel_total = 0

    for template_id in range(7, 11):
        print(f"Testing template {template_id}...")
        template_success = 0

        for i in range(10):
            obs = env.reset(template_id=template_id, seed=i)
            done = False
            episode_reward = 0

            for t in range(200):
                entities_t = torch.tensor(obs['entities'], dtype=torch.float32).unsqueeze(0).to(device)
                mask_t = torch.tensor(obs['entity_mask'], dtype=torch.bool).unsqueeze(0).to(device)

                with torch.no_grad():
                    action, _, _ = model.get_action(entities_t, mask_t, deterministic=True)

                obs, reward, done, info = env.step(action)
                episode_reward += reward

                if done:
                    break

            if info.get('success', False):
                template_success += 1
                novel_success += 1
            novel_total += 1

        print(f"  Template {template_id}: {template_success}/10 successes")

    novel_rate = novel_success / novel_total if novel_total > 0 else 0.0
    print(f"Novel Layouts Success Rate: {novel_rate:.2f} ({novel_success}/{novel_total})")

    print("\n=== PPO ABSTRACT RESULTS (RL-OPTIMIZED) ===")
    print(f"Training Layouts (T1-T6): {training_rate:.2f} ({training_success}/{training_total})")
    print(f"Novel Layouts (T7-T10): {novel_rate:.2f} ({novel_success}/{novel_total})")

    env.close()


if __name__ == "__main__":
    test_ppo_abstract()


