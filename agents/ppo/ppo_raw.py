"""
PPO Raw - Proximal Policy Optimization on pixel observations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
import sys
from tqdm import tqdm

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from envs.keydoor.keydoor_rl_env import KeyDoorRLEnv


class PPORaw(nn.Module):
    """
    PPO Raw - Proximal Policy Optimization on pixel observations.
    
    Architecture:
    - Conv encoder: Conv(1,32,3) -> ReLU -> Conv(32,64,3) -> ReLU -> AdaptiveAvgPool2d(4,4)
    - Actor: FC(64*4*4, 128) -> ReLU -> FC(128, 6) -> softmax
    - Critic: FC(64*4*4, 128) -> ReLU -> FC(128, 1)
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 max_grid_size: int = 12,
                 n_actions: int = 6,
                 hidden_dim: int = 128):
        super(PPORaw, self).__init__()
        
        self.input_channels = input_channels
        self.max_grid_size = max_grid_size
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        
        # Conv encoder
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Always reduce to 4x4 regardless of input size
        )
        
        # Calculate flattened size
        self.flattened_size = 64 * 4 * 4
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pixel_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pixel_obs: Pixel observations (batch_size, 1, H, W)
            
        Returns:
            Tuple of (action_logits, value)
        """
        # Conv encoding
        conv_out = self.conv_layers(pixel_obs)
        flattened = conv_out.view(conv_out.size(0), -1)
        
        # Actor and critic heads
        action_logits = self.actor(flattened)
        value = self.critic(flattened)
        
        return action_logits, value
    
    def get_action(self, pixel_obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Get action from policy.
        
        Args:
            pixel_obs: Pixel observation (1, 1, H, W)
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            action_logits, value = self.forward(pixel_obs)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1).item()
                log_prob = 0.0  # Deterministic action has no log prob
            else:
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action, device=action_logits.device)).item()
            
            return action, log_prob, value.item()


class PPORawTrainer:
    """
    PPO Raw Trainer for training PPO Raw models.
    """
    
    def __init__(self, 
                 model: PPORaw,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def collect_rollouts(self, 
                        env: KeyDoorRLEnv, 
                        n_steps: int,
                        template_range: Tuple[int, int] = (1, 7)) -> Dict[str, List]:
        """
        Collect rollouts from the environment.
        
        Args:
            env: Environment to collect rollouts from
            n_steps: Number of steps to collect
            template_range: Range of templates to use (start, end)
            
        Returns:
            Dictionary containing rollout data
        """
        self.model.eval()
        
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs = env.reset(template_id=np.random.randint(template_range[0], template_range[1]))
        episode_reward = 0
        
        for step in range(n_steps):
            # Convert observation to pixel format
            grid = obs['grid']
            pixel_obs = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get action from model
            action, log_prob, value = self.model.get_action(pixel_obs, deterministic=False)
            
            # Store data
            observations.append(pixel_obs.squeeze(0).cpu().numpy())
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            
            # Take step
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            
            # Reset if episode done
            if done:
                obs = env.reset(template_id=np.random.randint(template_range[0], template_range[1]))
                episode_reward = 0
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': log_probs,
            'dones': dones
        }
    
    def compute_gae(self, 
                   rewards: List[float], 
                   values: List[float], 
                   dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        # Compute returns and advantages
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, rollouts: Dict[str, List], epochs: int = 4) -> Dict[str, float]:
        """
        Update the model using PPO.
        
        Args:
            rollouts: Rollout data
            epochs: Number of update epochs
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        # Convert to tensors
        observations = torch.tensor(np.array(rollouts['observations']), dtype=torch.float32).to(self.device)
        # observations should already be in correct shape (batch_size, 1, H, W)
        # No reshaping needed since we stored them correctly
        actions = torch.tensor(rollouts['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(rollouts['log_probs'], dtype=torch.float32).to(self.device)
        advantages, returns = self.compute_gae(rollouts['rewards'], rollouts['values'], rollouts['dones'])
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(epochs):
            # Forward pass
            action_logits, values = self.model(observations)
            
            # Compute policy loss
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        return {
            'policy_loss': total_policy_loss / epochs,
            'value_loss': total_value_loss / epochs,
            'entropy': total_entropy / epochs,
            'total_loss': (total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy) / epochs
        }
    
    def train(self, 
              env: KeyDoorRLEnv, 
              total_steps: int = 100000,
              steps_per_update: int = 2048,
              epochs_per_update: int = 4,
              save_interval: int = 10000,
              output_dir: str = "experiments/exp_ppo_raw_rl_keydoor_seed0") -> None:
        """
        Train the PPO model.
        
        Args:
            env: Environment to train on
            total_steps: Total number of training steps
            steps_per_update: Number of steps per update
            epochs_per_update: Number of epochs per update
            save_interval: Save model every N steps
            output_dir: Output directory for saving models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"=== Training PPO Raw ===")
        print(f"Total steps: {total_steps}")
        print(f"Steps per update: {steps_per_update}")
        print(f"Epochs per update: {epochs_per_update}")
        print(f"Device: {self.device}")
        
        step_count = 0
        update_count = 0
        
        with tqdm(total=total_steps, desc="Training PPO Raw") as pbar:
            while step_count < total_steps:
                # Collect rollouts
                rollouts = self.collect_rollouts(env, steps_per_update)
                step_count += steps_per_update
                
                # Update model
                metrics = self.update(rollouts, epochs_per_update)
                update_count += 1
                
                # Update progress bar
                pbar.update(steps_per_update)
                pbar.set_postfix({
                    'Policy Loss': f"{metrics['policy_loss']:.4f}",
                    'Value Loss': f"{metrics['value_loss']:.4f}",
                    'Entropy': f"{metrics['entropy']:.4f}"
                })
                
                # Save model
                if step_count % save_interval == 0:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'step_count': step_count,
                        'update_count': update_count,
                        'metrics': metrics
                    }, os.path.join(output_dir, f'ppo_raw_step_{step_count}.pth'))
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': step_count,
            'update_count': update_count,
            'metrics': metrics
        }, os.path.join(output_dir, 'ppo_raw_final.pth'))
        
        print(f"✅ PPO Raw training completed!")
        print(f"Final metrics: {metrics}")
        print(f"Model saved to {output_dir}")