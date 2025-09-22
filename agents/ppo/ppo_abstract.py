"""
PPO Abstract - Proximal Policy Optimization on entity-list observations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import random


class PPOAbstract(nn.Module):
    """
    PPO policy/value network operating on entity-list observations with masks.

    Architecture:
    - Per-entity encoder: MLP(entity_dim -> hidden -> hidden)
    - Pooling: masked mean pooling to get global feature
    - Actor head: FC(hidden -> hidden -> n_actions)
    - Critic head: FC(hidden -> hidden -> 1)
    """

    def __init__(
        self,
        entity_dim: int = 10,
        max_entities: int = 16,
        n_actions: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.entity_dim = entity_dim
        self.max_entities = max_entities
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Per-entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor and Critic heads
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, max_entities, hidden)
        mask: (batch, max_entities) bool
        returns: (batch, hidden)
        """
        mask_f = mask.float().unsqueeze(-1)  # (batch, max_entities, 1)
        summed = (x * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, entities: torch.Tensor, entity_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        entities: (batch, max_entities, entity_dim)
        entity_mask: (batch, max_entities) bool
        returns: (action_logits, value)
        """
        bsz, max_entities, feat = entities.shape
        flat = entities.view(bsz * max_entities, feat)
        enc = self.entity_encoder(flat).view(bsz, max_entities, self.hidden_dim)
        pooled = self._masked_mean(enc, entity_mask)
        action_logits = self.actor(pooled)
        value = self.critic(pooled)
        return action_logits, value

    def get_action(
        self, entities: torch.Tensor, entity_mask: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Single-step action selection.
        Returns (action, log_prob, value)
        """
        with torch.no_grad():
            action_logits, value = self.forward(entities, entity_mask)
            if deterministic:
                action = torch.argmax(action_logits, dim=-1).item()
                log_prob = 0.0
            else:
                dist = torch.distributions.Categorical(logits=action_logits)
                action_t = dist.sample()
                action = action_t.item()
                log_prob = dist.log_prob(action_t).item()
            return action, log_prob, value.item()


class PPOAbstractTrainer:
    """
    PPO trainer mirroring the raw PPO pipeline, adapted for entity observations.
    """

    def __init__(
        self,
        model: PPOAbstract,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def collect_rollouts(
        self,
        env,
        n_steps: int,
        template_range: Tuple[int, int] = (1, 7),
        epsilon: float = 0.0,
    ) -> Dict[str, List]:
        """
        Collect rollouts using entity observations from env.
        The env must return obs with keys 'entities' and 'entity_mask'.
        """
        self.model.eval()

        entities_list = []
        masks_list = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        obs = env.reset(template_id=np.random.randint(template_range[0], template_range[1]))

        for _ in range(n_steps):
            ent = torch.tensor(obs["entities"], dtype=torch.float32).unsqueeze(0).to(self.device)
            mask = torch.tensor(obs["entity_mask"], dtype=torch.bool).unsqueeze(0).to(self.device)

            if random.random() < epsilon:
                # epsilon-greedy random action for exploration
                action = random.randrange(self.model.n_actions)
                with torch.no_grad():
                    logits, value_t = self.model(ent, mask)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(torch.tensor([action], device=self.device)).item()
                    value = value_t.item()
            else:
                action, log_prob, value = self.model.get_action(ent, mask, deterministic=False)

            entities_list.append(ent.squeeze(0).cpu().numpy())
            masks_list.append(mask.squeeze(0).cpu().numpy())
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)

            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            dones.append(done)

            if done:
                obs = env.reset(template_id=np.random.randint(template_range[0], template_range[1]))

        return {
            "entities": entities_list,
            "masks": masks_list,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "dones": dones,
        }

    def compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        advantages: List[float] = []
        returns: List[float] = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - float(dones[t])) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return advantages, returns

    def update(self, rollouts: Dict[str, List], epochs: int = 4) -> Dict[str, float]:
        self.model.train()

        entities = torch.tensor(np.array(rollouts["entities"]), dtype=torch.float32).to(self.device)
        masks = torch.tensor(np.array(rollouts["masks"]), dtype=torch.bool).to(self.device)
        actions = torch.tensor(rollouts["actions"], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(rollouts["log_probs"], dtype=torch.float32).to(self.device)
        advantages, returns = self.compute_gae(rollouts["rewards"], rollouts["values"], rollouts["dones"])
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(epochs):
            logits, values = self.model(entities, masks)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values.squeeze(), returns)
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        epochs_f = float(epochs)
        return {
            "policy_loss": total_policy_loss / epochs_f,
            "value_loss": total_value_loss / epochs_f,
            "entropy": total_entropy / epochs_f,
            "total_loss": (total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy)
            / epochs_f,
        }


