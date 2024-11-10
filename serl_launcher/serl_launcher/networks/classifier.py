import torch
import torch.nn as nn
from einops import rearrange


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        encoder: nn.Module,
        network: nn.Module,
        enable_stacking: bool = False
    ):
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        self.encoder = encoder
        self.network = network
        self.enable_stacking = enable_stacking
        self.output_layer = nn.Linear(network.output_dim, 1)

    def forward(self, x: torch.Tensor, train: bool = False, return_encoded: bool = False, 
                classify_encoded: bool = False) -> torch.Tensor:
        if return_encoded:
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(x.shape) == 4:
                    x = rearrange(x, 'T H W C -> H W (T C)')
                if len(x.shape) == 5:
                    x = rearrange(x, 'B T H W C -> B H W (T C)')
            x = self.pretrained_encoder(x)
            return x

        x = self.encoder(x, train=train, is_encoded=classify_encoded)
        x = self.network(x, train=train)
        x = self.output_layer(x).squeeze()
        return x
