from typing import Callable, Optional, Sequence
import torch
import torch.nn as nn

def default_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)

class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.activate_final = activate_final
        layers = []
        
        for i, size in enumerate(hidden_dims):
            layers.append(nn.Linear(hidden_dims[i-1] if i > 0 else hidden_dims[0], size))
            default_init()(layers[-1].weight)
            
            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(size))
                layers.append(activations if isinstance(activations, nn.Module) else getattr(nn, activations)())
                
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        self.train(train)
        return self.net(x)

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
        
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)
            
        self.dense1 = nn.Linear(features, features * 4)
        self.act = act
        self.dense2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        self.train(train)
        residual = x
        
        if self.dropout_rate is not None and self.dropout_rate > 0:
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
        activations: Callable = nn.SiLU()
    ):
        super().__init__()
        self.input_layer = nn.Linear(hidden_dim, hidden_dim)
        default_init()(self.input_layer.weight)
        
        self.blocks = nn.ModuleList([
            MLPResNetBlock(
                hidden_dim,
                act=activations,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate
            ) for _ in range(num_blocks)
        ])
        
        self.activations = activations
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        default_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        self.train(train)
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
