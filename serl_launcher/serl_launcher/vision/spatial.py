from typing import Sequence, Callable
import torch
import torch.nn as nn


class SpatialLearnedEmbeddings(nn.Module):
    """Spatial learned embeddings module that learns position-sensitive features."""
    
    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        num_features: int = 5,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        # Initialize learnable kernel parameter
        self.kernel = nn.Parameter(
            torch.empty(height, width, channel, num_features, dtype=dtype)
        )
        # Initialize using Lecun normal initialization
        nn.init.kaiming_normal_(self.kernel, mode='fan_in', nonlinearity='linear')

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process features through spatial learned embeddings.
        
        Args:
            features: Input tensor of shape [B x H x W x C] or [H x W x C]
            
        Returns:
            Processed features of shape [B x (C*num_features)] or [(C*num_features)]
        """
        squeeze = False
        if len(features.shape) == 3:
            features = features.unsqueeze(0)
            squeeze = True

        batch_size = features.shape[0]
        
        # Expand dimensions for broadcasting
        features = features.unsqueeze(-1)  # [B, H, W, C, 1]
        kernel = self.kernel.unsqueeze(0)   # [1, H, W, C, num_features]
        
        # Compute spatial embeddings
        features = (features * kernel).sum(dim=(1, 2))  # [B, C, num_features]
        features = features.reshape(batch_size, -1)     # [B, C*num_features]
        
        if squeeze:
            features = features.squeeze(0)
            
        return features
