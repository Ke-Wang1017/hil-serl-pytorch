from typing import Dict, Iterable, Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
        proprio_latent_dim: Dimension of proprioception latent space.
        enable_stacking: Whether to enable frame stacking.
        image_keys: Keys for image observations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        use_proprio: bool,
        proprio_latent_dim: int = 64,
        enable_stacking: bool = False,
        image_keys: Iterable[str] = ("image",)
    ):
        super().__init__()
        self.encoder = encoder
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim
        self.enable_stacking = enable_stacking
        self.image_keys = image_keys

        if use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(
                    self.proprio_latent_dim, 
                    self.proprio_latent_dim, 
                    bias=True
                ),
                nn.LayerNorm(self.proprio_latent_dim),
                nn.Tanh()
            )
            # Initialize the linear layer
            nn.init.xavier_uniform_(self.proprio_encoder[0].weight)
            nn.init.zeros_(self.proprio_encoder[0].bias)

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> torch.Tensor:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, 'T H W C -> H W (T C)')
                    if len(image.shape) == 5:
                        image = rearrange(image, 'B T H W C -> B H W (T C)')

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = image.detach()

            encoded.append(image)

        encoded = torch.cat(encoded, dim=-1)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, 'T C -> (T C)')
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, 'B T C -> B (T C)')
            
            state = self.proprio_encoder(state)
            encoded = torch.cat([encoded, state], dim=-1)

        return encoded
