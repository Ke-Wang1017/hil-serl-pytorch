from typing import Callable, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_init(scale=1.0):
    return lambda x: nn.init.orthogonal_(x, gain=scale)

class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = F.silu,
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.activate_final = activate_final
        self.activations = activations if not isinstance(activations, str) else getattr(F, activations)
        
        layers = []
        prev_dim = None
        for i, size in enumerate(hidden_dims):
            if prev_dim is not None:
                layers.append(nn.Linear(prev_dim, size))
                nn.init.orthogonal_(layers[-1].weight, gain=1.0)
                nn.init.zeros_(layers[-1].bias)
                
                if i + 1 < len(hidden_dims) or activate_final:
                    if dropout_rate is not None and dropout_rate > 0:
                        layers.append(nn.Dropout(p=dropout_rate))
                    if use_layer_norm:
                        layers.append(nn.LayerNorm(size))
                    layers.append(nn.Module())  # Placeholder for activation
            prev_dim = size
            
        self.layers = nn.ModuleList(layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                x = layer(x) if train else x
            elif isinstance(layer, nn.Module) and not isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Dropout)):
                x = self.activations(x)
            else:
                x = layer(x)
        return x

class MLPResNetBlock(nn.Module):
    def __init__(
        self,
        features: int,
        act: Callable,
        dropout_rate: float = None,
        use_layer_norm: bool = False
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.act = act
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
            
        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)
        
        # Initialize weights
        for layer in [self.dense1, self.dense2, self.residual]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        residual = x
        
        if hasattr(self, 'dropout') and train:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
            
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)
        
        if residual.shape != x.shape:
            residual = self.residual(residual)
            
        return residual + x

class MLPResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        out_dim: int,
        dropout_rate: float = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activations: Callable = F.silu
    ):
        super().__init__()
        self.activations = activations
        
        self.input_layer = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.input_layer.weight, gain=1.0)
        nn.init.zeros_(self.input_layer.bias)
        
        self.blocks = nn.ModuleList([
            MLPResNetBlock(
                hidden_dim,
                act=activations,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate
            ) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        nn.init.orthogonal_(self.output_layer.weight, gain=1.0)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, train=train)
        x = self.activations(x)
        x = self.output_layer(x)
        return x

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(init_value))

    def forward(self):
        return self.value
