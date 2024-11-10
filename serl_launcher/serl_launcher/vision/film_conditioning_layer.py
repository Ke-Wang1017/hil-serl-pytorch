# adapted from https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py
import torch
import torch.nn as nn


class FilmConditioning(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_layer = None
        self.mult_layer = None

    def forward(self, conv_filters: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Applies FiLM conditioning to a convolutional feature map.

        Args:
            conv_filters: A tensor of shape [batch_size, channels, height, width].
            conditioning: A tensor of shape [batch_size, conditioning_size].

        Returns:
            A tensor of shape [batch_size, channels, height, width].
        """
        # Initialize layers on first forward pass to get correct sizes
        if self.add_layer is None:
            self.add_layer = nn.Linear(
                conditioning.shape[-1],
                conv_filters.shape[1],  # number of channels
                bias=True
            )
            self.mult_layer = nn.Linear(
                conditioning.shape[-1],
                conv_filters.shape[1],  # number of channels
                bias=True
            )
            
            # Initialize weights and biases to zero
            nn.init.zeros_(self.add_layer.weight)
            nn.init.zeros_(self.add_layer.bias)
            nn.init.zeros_(self.mult_layer.weight)
            nn.init.zeros_(self.mult_layer.bias)

        projected_cond_add = self.add_layer(conditioning)
        projected_cond_mult = self.mult_layer(conditioning)

        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1]
        projected_cond_add = projected_cond_add.unsqueeze(-1).unsqueeze(-1)
        projected_cond_mult = projected_cond_mult.unsqueeze(-1).unsqueeze(-1)

        return conv_filters * (1 + projected_cond_add) + projected_cond_mult


if __name__ == "__main__":
    # Test the implementation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create random input tensors
    x = torch.randn(1, 3, 32, 32, device=device)  # [B, C, H, W] format for PyTorch
    z = torch.ones(1, 64, device=device)
    
    # Initialize and test the model
    film = FilmConditioning().to(device)
    y = film(x, z)
    
    print(y.shape)  # Should print: torch.Size([1, 3, 32, 32])
