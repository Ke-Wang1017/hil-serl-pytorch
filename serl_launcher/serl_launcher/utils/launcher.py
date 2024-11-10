#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agentlace.trainer import TrainerConfig

from serl_launcher.common.typing import Batch
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.vision.data_augmentations import batched_random_crop

##############################################################################

def make_bc_agent(
    seed: int, 
    sample_obs: dict, 
    sample_action: np.ndarray, 
    image_keys: tuple = ("image",), 
    encoder_type: str = "resnet-pretrained",
    device: torch.device = None,
):
    """Create a BC agent."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    torch.manual_seed(seed)
    
    return BCAgent.create(
        device,
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )

def make_sac_pixel_agent(
    seed: int,
    sample_obs: dict,
    sample_action: np.ndarray,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet-pretrained",
    reward_bias: float = 0.0,
    target_entropy: float = None,
    discount: float = 0.97,
    device: torch.device = None,
):
    """Create a SAC agent for pixel observations."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    torch.manual_seed(seed)
    
    agent = SACAgent.create_pixels(
        device,
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent

def make_sac_pixel_agent_hybrid_single_arm(
    seed: int,
    sample_obs: dict,
    sample_action: np.ndarray,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet-pretrained",
    reward_bias: float = 0.0,
    target_entropy: float = None,
    discount: float = 0.97,
    device: torch.device = None,
):
    """Create a SAC agent for single arm with hybrid policy."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    torch.manual_seed(seed)
    
    agent = SACAgentHybridSingleArm.create_pixels(
        device,
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent

def make_sac_pixel_agent_hybrid_dual_arm(
    seed: int,
    sample_obs: dict,
    sample_action: np.ndarray,
    image_keys: tuple = ("image",),
    encoder_type: str = "resnet-pretrained",
    reward_bias: float = 0.0,
    target_entropy: float = None,
    discount: float = 0.97,
    device: torch.device = None,
):
    """Create a SAC agent for dual arm with hybrid policy."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    torch.manual_seed(seed)
    
    agent = SACAgentHybridDualArm.create_pixels(
        device,
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": F.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent

def linear_schedule(step: int) -> float:
    """Linear learning rate schedule."""
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000

    linear_step = min(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value

def make_batch_augmentation_func(image_keys: tuple) -> callable:
    """Create batch augmentation function."""
    def data_augmentation_fn(observations: dict) -> dict:
        for pixel_key in image_keys:
            observations = observations.copy()
            observations[pixel_key] = batched_random_crop(
                observations[pixel_key], padding=4, num_batch_dims=2
            )
        return observations
    
    def augment_batch(batch: Batch) -> Batch:
        obs = data_augmentation_fn(batch["observations"])
        next_obs = data_augmentation_fn(batch["next_observations"])
        batch = batch.copy()
        batch["observations"] = obs
        batch["next_observations"] = next_obs
        return batch
    
    return augment_batch

def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589) -> TrainerConfig:
    """Create trainer configuration."""
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )

def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
) -> WandBLogger:
    """Create Weights & Biases logger."""
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
