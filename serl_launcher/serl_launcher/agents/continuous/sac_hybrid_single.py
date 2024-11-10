from functools import partial
from typing import Iterable, Optional, Tuple, Set
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch
from serl_launcher.serl_launcher.networks.actor_critic_nets import Critic, Policy, GraspCritic, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    target_params: Optional[dict] = None
    
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'target_params': self.target_params,
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.target_params = state_dict['target_params']
        
    def update_target_params(self, tau: float):
        """Soft update target parameters."""
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.target_params.values()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class SACAgentHybridSingleArm:
    def __init__(self, state: TrainState, config: dict):
        self.state = state
        self.config = config
        self.device = next(state.model.parameters()).device

    def forward_critic(
        self,
        observations: dict,
        actions: torch.Tensor,
        train: bool = True,
        target: bool = False,
    ) -> torch.Tensor:
        params = self.state.target_params if target else None
        with torch.set_grad_enabled(train and not target):
            return self.state.model(
                observations,
                actions,
                name="critic",
                train=train,
                params=params,
            )

    def forward_grasp_critic(
        self,
        observations: dict,
        train: bool = True,
        target: bool = False,
    ) -> torch.Tensor:
        params = self.state.target_params if target else None
        with torch.set_grad_enabled(train and not target):
            return self.state.model(
                observations,
                name="grasp_critic",
                train=train,
                params=params,
            )

    def forward_policy(
        self,
        observations: dict,
        train: bool = True,
    ) -> torch.distributions.Distribution:
        with torch.set_grad_enabled(train):
            return self.state.model(
                observations,
                name="actor",
                train=train,
            )

    def forward_temperature(self) -> torch.Tensor:
        return self.state.model(name="temperature")

    def temperature_lagrange_penalty(self, entropy: torch.Tensor) -> torch.Tensor:
        return self.state.model(
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch: dict):
        next_action_distributions = self.forward_policy(batch["next_observations"])
        next_actions = next_action_distributions.rsample()
        next_actions_log_probs = next_action_distributions.log_prob(next_actions)
        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        batch_size = batch["rewards"].shape[0]
        # Extract continuous actions for critic
        actions = batch["actions"][..., :-1]  # Remove grasp action

        next_actions, next_actions_log_probs = self._compute_next_actions(batch)

        # Evaluate next Qs for all ensemble members
        target_next_qs = self.forward_critic(
            batch["next_observations"],
            next_actions,
            train=False,
            target=True,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            subsample_idcs = torch.randint(
                0, self.config["critic_ensemble_size"],
                (self.config["critic_subsample_size"],),
                device=self.device
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(dim=0)[0]

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            train=True,
        )

        target_qs = target_q.unsqueeze(0).expand(self.config["critic_ensemble_size"], -1)
        critic_loss = torch.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss.item(),
            "predicted_qs": predicted_qs.mean().item(),
            "target_qs": target_qs.mean().item(),
            "rewards": batch["rewards"].mean().item(),
        }

        return critic_loss, info

    def grasp_critic_loss_fn(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        batch_size = batch["rewards"].shape[0]
        grasp_action = (batch["actions"][..., -1]).to(torch.int64) + 1  # Cast to {0, 1, 2}
        
        # Forward passes
        target_next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            train=False,
            target=True,
        )
        
        next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            train=False,
        )
        best_next_grasp_action = next_grasp_qs.argmax(dim=-1)
        
        # Select target Q-values
        target_next_grasp_q = target_next_grasp_qs[torch.arange(batch_size), best_next_grasp_action]
        
        # Compute target Q-values
        grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
        target_grasp_q = (
            grasp_rewards
            + self.config["discount"] * batch["masks"] * target_next_grasp_q
        )
        
        # Get predicted Q-values
        predicted_grasp_qs = self.forward_grasp_critic(
            batch["observations"],
            train=True,
        )
        predicted_grasp_q = predicted_grasp_qs[torch.arange(batch_size), grasp_action]
        
        grasp_critic_loss = torch.mean((predicted_grasp_q - target_grasp_q) ** 2)
        
        info = {
            "grasp_critic_loss": grasp_critic_loss.item(),
            "predicted_grasp_qs": predicted_grasp_q.mean().item(),
            "target_grasp_qs": target_grasp_q.mean().item(),
            "grasp_rewards": grasp_rewards.mean().item(),
        }

        return grasp_critic_loss, info

    def policy_loss_fn(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        temperature = self.forward_temperature()

        action_distributions = self.forward_policy(
            batch["observations"],
            train=True,
        )
        actions = action_distributions.rsample()
        log_probs = action_distributions.log_prob(actions)

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            train=False,
        )
        predicted_q = predicted_qs.mean(dim=0)

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -torch.mean(actor_objective)

        info = {
            "actor_loss": actor_loss.item(),
            "temperature": temperature.item(),
            "entropy": -log_probs.mean().item(),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        _, next_actions_log_probs = self._compute_next_actions(batch)
        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(entropy)
        
        return temperature_loss, {"temperature_loss": temperature_loss.item()}

    def update(
        self,
        batch: Batch,
        networks_to_update: Set[str] = {"actor", "critic", "grasp_critic", "temperature"}
    ) -> Tuple[dict, dict]:
        batch_size = batch["rewards"].shape[0]
        assert batch["actions"].shape == (batch_size, 7)  # 6 continuous + 1 grasp

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        if "augmentation_function" in self.config and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch)

        # Add reward bias
        batch["rewards"] = batch["rewards"] + self.config["reward_bias"]

        # Move batch to device
        batch = {k: torch.tensor(v, device=self.device) if isinstance(v, np.ndarray) else v 
                for k, v in batch.items()}

        info = {}
        
        # Update networks
        self.state.model.train()
        for network in networks_to_update:
            self.state.optimizer.zero_grad()
            
            if network == "critic":
                loss, net_info = self.critic_loss_fn(batch)
            elif network == "grasp_critic":
                loss, net_info = self.grasp_critic_loss_fn(batch)
            elif network == "actor":
                loss, net_info = self.policy_loss_fn(batch)
            elif network == "temperature":
                loss, net_info = self.temperature_loss_fn(batch)
                
            loss.backward()
            self.state.optimizer.step()
            info.update(net_info)

        # Update target network if critic was updated
        if "critic" in networks_to_update:
            self.state.update_target_params(self.config["soft_target_update_rate"])

        return info

    def sample_actions(
        self,
        observations: dict,
        argmax: bool = False,
    ) -> np.ndarray:
        """Sample actions from policy."""
        self.state.model.eval()
        with torch.no_grad():
            observations = {k: torch.tensor(v, device=self.device) 
                          for k, v in observations.items()}
            
            # Sample continuous actions
            dist = self.forward_policy(observations, train=False)
            ee_actions = dist.mode() if argmax else dist.sample()
            
            # Get grasp actions
            grasp_q_values = self.forward_grasp_critic(
                observations,
                train=False,
            )
            grasp_action = grasp_q_values.argmax(dim=-1) - 1  # Map back to {-1, 0, 1}
            
            # Combine actions
            actions = torch.cat([
                ee_actions,
                grasp_action.unsqueeze(-1)
            ], dim=-1)
            
            return actions.cpu().numpy()

    @classmethod
    def create(cls, *args, **kwargs):
        """Factory method to create agent. Implementation similar to JAX version but with PyTorch."""
        # Implementation would be similar to the JAX version but using PyTorch components
        pass

    @classmethod
    def create_pixels(cls, *args, **kwargs):
        """Factory method to create pixel-based agent. Implementation similar to JAX version but with PyTorch."""
        # Implementation would be similar to the JAX version but using PyTorch components
        pass
