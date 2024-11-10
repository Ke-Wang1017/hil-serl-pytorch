from functools import partial
from typing import Any, Iterable, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch
from serl_launcher.serl_launcher.networks.actor_critic_nets import Policy
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop


@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    target_params: Optional[dict] = None
    rng: Optional[torch.Generator] = None
    
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


class BCAgent:
    def __init__(self, state: TrainState, config: dict):
        self.state = state
        self.config = config
        self.device = next(state.model.parameters()).device

    def data_augmentation_fn(self, observations):
        for pixel_key in self.config["image_keys"]:
            observations = observations.copy()
            observations[pixel_key] = batched_random_crop(
                observations[pixel_key], padding=4, num_batch_dims=2
            )
        return observations

    def update(self, batch: Batch):
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        if "augmentation_function" in self.config and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch)

        self.state.model.train()
        self.state.optimizer.zero_grad()

        # Move batch to device
        observations = {k: torch.tensor(v, device=self.device) for k, v in batch["observations"].items()}
        actions = torch.tensor(batch["actions"], device=self.device)

        # Forward pass
        dist = self.state.model(
            observations,
            temperature=1.0,
            train=True,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(actions)
        mse = ((pi_actions - actions) ** 2).sum(-1)
        actor_loss = -(log_probs).mean()

        # Backward pass and optimize
        actor_loss.backward()
        self.state.optimizer.step()

        info = {
            "actor_loss": actor_loss.item(),
            "mse": mse.mean().item(),
        }

        return info
    
    def forward_policy(self, observations: dict, *, temperature: float = 1.0, non_squash_distribution: bool = False):
        self.state.model.eval()
        with torch.no_grad():
            observations = {k: torch.tensor(v, device=self.device) for k, v in observations.items()}
            dist = self.state.model(
                observations,
                train=False,
                temperature=temperature,
                name="actor",
                non_squash_distribution=non_squash_distribution
            )
        return dist

    def sample_actions(
        self,
        observations: dict,
        *,
        temperature: float = 1.0,
        argmax: bool = False,
    ) -> torch.Tensor:
        self.state.model.eval()
        with torch.no_grad():
            observations = {k: torch.tensor(v, device=self.device) for k, v in observations.items()}
            dist = self.state.model(
                observations,
                temperature=temperature,
                name="actor",
            )
            if argmax:
                actions = dist.mode()
            else:
                actions = dist.sample()
        return actions.cpu().numpy()

    def get_debug_metrics(self, batch):
        self.state.model.eval()
        with torch.no_grad():
            observations = {k: torch.tensor(v, device=self.device) for k, v in batch["observations"].items()}
            actions = torch.tensor(batch["actions"], device=self.device)
            
            dist = self.state.model(
                observations,
                temperature=1.0,
                name="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(actions)
            mse = ((pi_actions - actions) ** 2).sum(-1)

        return {
            "mse": mse.cpu().numpy(),
            "log_probs": log_probs.cpu().numpy(),
            "pi_actions": pi_actions.cpu().numpy(),
        }

    @classmethod
    def create(
        cls,
        device: torch.device,
        observations: dict,
        actions: np.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        image_keys: Iterable[str] = ("image",),
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        augmentation_function: Optional[callable] = None,
    ):
        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        network_kwargs["activate_final"] = True
        network = MLP(**network_kwargs)
        
        model = Policy(
            encoder_def,
            network,
            action_dim=actions.shape[-1],
            **policy_kwargs,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        state = TrainState(
            model=model,
            optimizer=optimizer,
            target_params=model.state_dict().copy(),
            rng=torch.Generator(device=device),
        )

        config = dict(
            image_keys=image_keys,
            augmentation_function=augmentation_function
        )

        agent = cls(state, config)

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
