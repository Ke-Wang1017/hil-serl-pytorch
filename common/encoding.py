from typing import Dict, Iterable, Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange
from functools import lru_cache

"""
# Create wrapper with optimizations
wrapper = EncodingWrapper(
    encoder={"image": image_encoder},
    use_proprio=True,
    device="cuda"
)

# Forward pass with automatic mixed precision
with torch.cuda.amp.autocast():
    encoded = wrapper(
        observations={
            "image": images.to(device, non_blocking=True),
            "state": states.to(device, non_blocking=True)
        },
        train=True
    )
"""

class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, with proprioception and gradient control.
    Optimized for better memory usage and faster execution.
    """

    def __init__(
        self,
        encoder: Dict[str, nn.Module],
        use_proprio: bool,
        proprio_latent_dim: int = 64,
        enable_stacking: bool = False,
        image_keys: Iterable[str] = ("image",),
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        if encoder is None:
            self._use_image_encoder = False
        else:
            self._use_image_encoder = True
            self.encoder = nn.ModuleDict(encoder)
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim
        self.enable_stacking = enable_stacking
        self.image_keys = tuple(image_keys)  # Convert to tuple for hashable caching
        
        if self.use_proprio:
            self.proprio_encoder = None  # Will be initialized lazily
            self._proprio_initialized = False
        
        self.to(self.device)

    @torch.jit.unused  # Disable TorchScript for this method
    def _init_proprio_encoder(self, input_dim: int) -> None:
        """Lazy initialization of proprioceptive encoder"""
        self.proprio_encoder = nn.Sequential(
            nn.Linear(input_dim, self.proprio_latent_dim),
            nn.LayerNorm(self.proprio_latent_dim),
            nn.Tanh()
        ).to(self.device)
        self._proprio_initialized = True

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_rearrange_pattern(shape: Tuple[int, ...], is_nchw: bool) -> str:
        """Cache rearrange patterns for common input shapes"""
        if len(shape) == 4:
            return 'T C H W -> H W (T C)' if is_nchw else 'T H W C -> H W (T C)'
        elif len(shape) == 5:
            return 'B T C H W -> B H W (T C)' if is_nchw else 'B T H W C -> B H W (T C)'
        return ''

    def _process_image(self, image: torch.Tensor, is_encoded: bool) -> torch.Tensor:
        """Process a single image tensor"""
        if not is_encoded and self.enable_stacking:
            # Determine format and get cached pattern
            is_nchw = image.shape[1] in [1, 3] if len(image.shape) == 4 else image.shape[2] in [1, 3]
            pattern = self._get_rearrange_pattern(image.shape, is_nchw)
            if pattern:
                image = rearrange(image, pattern)
        return image

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> torch.Tensor:
        self.train(train)

        # Pre-process observations
        observations = {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
            for k, v in observations.items()
        }

        # Process images
        encoded = []
        if self._use_image_encoder:
            for image_key in self.image_keys:
                image = self._process_image(observations[image_key], is_encoded)
                
                # Encode image with gradient control
                with torch.set_grad_enabled(not stop_gradient):
                    image = self.encoder[image_key](
                        image, 
                        train=train, 
                        encode=not is_encoded
                    )
                encoded.append(image)
            # Concatenate encoded images
            encoded = torch.cat(encoded, dim=-1)

        # Process proprioceptive state if needed
        if self.use_proprio:
            state = observations["state"]
            # Handle stacking for state
            if self.enable_stacking:
                if state.dim() == 2:
                    state = rearrange(state, 'T C -> (T C)')
                    if self._use_image_encoder:
                        encoded = encoded.reshape(-1)
                elif state.dim() == 3:
                    state = rearrange(state, 'B T C -> B (T C)')

            # Lazy initialization of proprio encoder
            if not self._proprio_initialized:
                self._init_proprio_encoder(state.shape[-1])

            # Encode and concatenate state
            state = self.proprio_encoder(state)
            if self._use_image_encoder:
                encoded = torch.cat([encoded, state], dim=-1)
            else:
                encoded = state

        return encoded

    def __repr__(self) -> str:
        """Enhanced string representation for debugging"""
        encoder_keys = list(self.encoder.keys()) if self._use_image_encoder else []
        return (f"EncodingWrapper(encoder_keys={encoder_keys}, "
                f"use_proprio={self.use_proprio}, "
                f"proprio_latent_dim={self.proprio_latent_dim}, "
                f"enable_stacking={self.enable_stacking}, "
                f"device={self.device})")
