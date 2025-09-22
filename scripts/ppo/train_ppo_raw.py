#!/usr/bin/env python3
"""
Train PPO Raw model using RL-optimized KeyDoor environment with exploration fixes.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import argparse
import os
from tqdm import tqdm
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from envs.keydoor.keydoor_rl_env import KeyDoorRLEnv
from agents.ppo.ppo_raw import PPORaw, PPORawTrainer

def train_ppo_raw(
    total_steps: int = 100000,
    steps_per_update: int = 2048,
    epochs_per_update: int = 4,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    entropy_warmup: float = 0.05,
    warmup_steps: int = 15000,
    epsilon: float = 0.05,
    curriculum_steps: int = 30000,
    max_steps: int = 200,
    save_interval: int = 10000,
    output_dir: str = "experiments/exp_ppo_raw_rl_keydoor_seed0"
):
    """Train PPO Raw model with exploration fixes and curriculum learning."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = PPORaw(
        input_channels=1,
        max_grid_size=12,
        n_actions=6,
        hidden_dim=128
    )
    
    # Create trainer
    trainer = PPORawTrainer(
        model=model,
        learning_rate=lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=0.5
    )
    
    # Create environment
    env = KeyDoorRLEnv(max_steps=max_steps)
    
    # Training loop
    step_count = 0
    update_count = 0
    
    # Progress bar
    pbar = tqdm(total=total_steps, desc="Training PPO Raw")
    
    while step_count < total_steps:
        # Curriculum learning: start with T1-T3, then expand to T1-T6
        if step_count < curriculum_steps:
            template_range = (1, 4)  # T1-T3
        else:
            template_range = (1, 7)  # T1-T6
        
        # Entropy schedule
        if step_count < warmup_steps:
            frac = step_count / max(1, warmup_steps)
            trainer.entropy_coef = entropy_warmup + (entropy_coef - entropy_warmup) * frac
        else:
            trainer.entropy_coef = entropy_coef
        
        # Collect rollouts
        rollouts = trainer.collect_rollouts(
            env=env,
            n_steps=steps_per_update,
            template_range=template_range
        )
        
        # Update model
        metrics = trainer.update(rollouts, epochs_per_update)
        
        # Update counters
        step_count += steps_per_update
        update_count += 1
        
        # Update progress bar
        pbar.update(steps_per_update)
        pbar.set_postfix({
            "PolicyLoss": f"{metrics['policy_loss']:.4f}",
            "ValueLoss": f"{metrics['value_loss']:.4f}",
            "Entropy": f"{metrics['entropy']:.4f}",
            "EntCoef": f"{trainer.entropy_coef:.3f}",
            "Eps": f"{epsilon:.2f}",
            "Tpl": f"{template_range[0]}-{template_range[1]-1}"
        })
        
        # Save model
        if step_count % save_interval == 0:
            save_path = os.path.join(output_dir, f"ppo_raw_step_{step_count}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "step_count": step_count,
                "update_count": update_count,
                "metrics": metrics,
                "config": {
                    "total_steps": total_steps,
                    "steps_per_update": steps_per_update,
                    "epochs_per_update": epochs_per_update,
                    "lr": lr,
                    "clip_ratio": clip_ratio,
                    "value_coef": value_coef,
                    "entropy_coef": entropy_coef,
                    "entropy_warmup": entropy_warmup,
                    "warmup_steps": warmup_steps,
                    "epsilon": epsilon,
                    "curriculum_steps": curriculum_steps,
                    "max_steps": max_steps
                }
            }, save_path)
            print(f"\nModel saved to {save_path}")
    
    # Save final model
    final_save_path = os.path.join(output_dir, "ppo_raw_final.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "step_count": step_count,
        "update_count": update_count,
        "metrics": metrics,
        "config": {
            "total_steps": total_steps,
            "steps_per_update": steps_per_update,
            "epochs_per_update": epochs_per_update,
            "lr": lr,
            "clip_ratio": clip_ratio,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "entropy_warmup": entropy_warmup,
            "warmup_steps": warmup_steps,
            "epsilon": epsilon,
            "curriculum_steps": curriculum_steps,
            "max_steps": max_steps
        }
    }, final_save_path)
    
    pbar.close()
    print(f"\nPPO Raw training completed!")
    print(f"Final model saved to {final_save_path}")
    print(f"Total steps: {step_count}")
    print(f"Total updates: {update_count}")
    print(f"Final metrics: {metrics}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO Raw model with RL-optimized environment.")
    parser.add_argument('--output_dir', type=str, default='experiments/exp_ppo_raw_rl_keydoor_seed0',
                        help='Output directory for the model')
    parser.add_argument('--total_steps', type=int, default=100000,
                        help='Total number of training steps')
    parser.add_argument('--steps_per_update', type=int, default=2048,
                        help='Number of steps per update')
    parser.add_argument('--epochs_per_update', type=int, default=4,
                        help='Number of epochs per update')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='PPO clip ratio')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--entropy_warmup', type=float, default=0.05,
                        help='Initial entropy coefficient')
    parser.add_argument('--warmup_steps', type=int, default=15000,
                        help='Entropy warmup steps')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Epsilon-greedy exploration')
    parser.add_argument('--curriculum_steps', type=int, default=30000,
                        help='Curriculum learning steps')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Max steps per episode')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='Save interval')
    args = parser.parse_args()

    train_ppo_raw(**vars(args))

if __name__ == "__main__":
    main()

