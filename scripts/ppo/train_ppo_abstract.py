#!/usr/bin/env python3
"""
Train PPO Abstract model using entity observations from KeyDoorRLEnv
"""

import torch
import numpy as np
from pathlib import Path
import sys
import argparse
import os
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from envs.keydoor.keydoor_rl_env import KeyDoorRLEnv
from agents.ppo.ppo_abstract import PPOAbstract, PPOAbstractTrainer


def main():
    parser = argparse.ArgumentParser(description="Train PPO Abstract model with entity observations.")
    parser.add_argument('--output_dir', type=str, default='experiments/exp_ppo_abstract_rl_keydoor_seed0',
                        help='Output directory for the model')
    parser.add_argument('--total_steps', type=int, default=100000,
                        help='Total number of training steps')
    parser.add_argument('--steps_per_update', type=int, default=2048,
                        help='Number of steps per update')
    parser.add_argument('--epochs_per_update', type=int, default=4,
                        help='Number of epochs per update')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='Save model every N steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='PPO clip ratio')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient (final, used after warmup)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum gradient norm')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Epsilon-greedy exploration probability during rollout collection')
    parser.add_argument('--entropy_warmup', type=float, default=0.05,
                        help='Initial entropy coefficient for warmup')
    parser.add_argument('--warmup_steps', type=int, default=30000,
                        help='Number of steps over which to decay entropy from warmup to final')
    parser.add_argument('--curriculum_steps', type=int, default=30000,
                        help='Steps to train on templates 1-3 before widening to 1-6')
    args = parser.parse_args()

    # Create environment
    env = KeyDoorRLEnv(max_steps=200)

    # Create model
    model = PPOAbstract(
        entity_dim=10,
        max_entities=16,
        n_actions=6,
        hidden_dim=256,
    )

    # Create trainer
    trainer = PPOAbstractTrainer(
        model=model,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )

    # Train loop (mirrors PPO Raw script style)
    os.makedirs(args.output_dir, exist_ok=True)
    step_count = 0
    update_count = 0

    print("=== Training PPO Abstract ===")
    print(f"Total steps: {args.total_steps}")
    print(f"Steps per update: {args.steps_per_update}")
    print(f"Epochs per update: {args.epochs_per_update}")
    print(f"Device: {trainer.device}")

    with tqdm(total=args.total_steps, desc="Training PPO Abstract") as pbar:
        while step_count < args.total_steps:
            # Curriculum: templates 1-3, then 1-6
            if step_count < args.curriculum_steps:
                template_range = (1, 4)  # 1-3 inclusive
            else:
                template_range = (1, 7)  # 1-6 inclusive

            # Entropy schedule: linear decay from warmup -> final over warmup_steps
            if step_count < args.warmup_steps:
                frac = step_count / max(1, args.warmup_steps)
                trainer.entropy_coef = args.entropy_warmup + (args.entropy_coef - args.entropy_warmup) * frac
            else:
                trainer.entropy_coef = args.entropy_coef

            rollouts = trainer.collect_rollouts(env, args.steps_per_update, template_range=template_range, epsilon=args.epsilon)
            step_count += args.steps_per_update

            metrics = trainer.update(rollouts, args.epochs_per_update)
            update_count += 1

            pbar.update(args.steps_per_update)
            pbar.set_postfix({
                'Policy Loss': f"{metrics['policy_loss']:.4f}",
                'Value Loss': f"{metrics['value_loss']:.4f}",
                'Entropy': f"{metrics['entropy']:.4f}",
                'EntCoef': f"{trainer.entropy_coef:.3f}",
                'Eps': f"{args.epsilon:.2f}",
                'Tpl': f"{template_range[0]}-{template_range[1]-1}"
            })

            # Save periodic checkpoints
            if step_count % args.save_interval == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'step_count': step_count,
                    'update_count': update_count,
                    'metrics': metrics
                }, os.path.join(args.output_dir, f'ppo_abstract_step_{step_count}.pth'))

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'step_count': step_count,
        'update_count': update_count,
        'metrics': metrics
    }, os.path.join(args.output_dir, 'ppo_abstract_final.pth'))

    print("PPO Abstract training completed!")
    print(f"Final metrics: {metrics}")
    print(f"Model saved to {args.output_dir}")

    env.close()


if __name__ == "__main__":
    main()
