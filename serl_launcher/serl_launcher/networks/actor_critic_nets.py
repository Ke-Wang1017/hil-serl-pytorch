from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.distributions as D
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

def default_init(scale=1.0):
    return partial(torch.nn.init.orthogonal_, gain=scale)

class ValueCritic(nn.Module):
    def __init__(self, encoder, network, init_final=None):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        self.output_layer = nn.Linear(network.output_dim, 1)
        if init_final is not None:
            torch.nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            torch.nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            default_init()(self.output_layer.weight)
            torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        outputs = self.network(self.encoder(observations))
        value = self.output_layer(outputs)
        return value.squeeze(-1)

class Critic(nn.Module):
    def __init__(self, encoder, network, init_final=None):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        self.output_layer = nn.Linear(network.output_dim, 1)
        if init_final is not None:
            torch.nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            torch.nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            default_init()(self.output_layer.weight)
            torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, train: bool = False) -> torch.Tensor:
        if len(actions.shape) == 3:  # Handle multiple actions per observation
            B, N, A = actions.shape
            observations = observations.unsqueeze(1).expand(-1, N, -1)
            observations = observations.reshape(B * N, -1)
            actions = actions.reshape(B * N, A)
            q_values = self._forward_single(observations, actions)
            return q_values.reshape(B, N)
        else:
            return self._forward_single(observations, actions)

    def _forward_single(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            obs_enc = self.encoder(observations)
        else:
            obs_enc = observations
            
        inputs = torch.cat([obs_enc, actions], dim=-1)
        outputs = self.network(inputs)
        value = self.output_layer(outputs)
        return value.squeeze(-1)

class GraspCritic(nn.Module):
    def __init__(self, encoder, network, init_final=None, output_dim=3):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        self.output_dim = output_dim
        self.output_layer = nn.Linear(network.output_dim, output_dim)
        if init_final is not None:
            torch.nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            torch.nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            default_init()(self.output_layer.weight)
            torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        if self.encoder is not None:
            obs_enc = self.encoder(observations)
        else:
            obs_enc = observations
            
        outputs = self.network(obs_enc)
        value = self.output_layer(outputs)
        return value  # (batch_size, output_dim)

class TanhNormal(TransformedDistribution):
    def __init__(self, loc, scale, low=None, high=None):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        
        # Initialize normal distribution
        normal = Normal(loc, scale)
        
        transforms = []
        
        if low is not None and high is not None:
            self.low_tensor = torch.tensor(low, device=loc.device)
            self.high_tensor = torch.tensor(high, device=loc.device)
            transforms.append(TanhTransform())
            transforms.append(lambda x: x * (high - low) / 2 + (high + low) / 2)
        else:
            transforms = [TanhTransform()]
            
        super().__init__(normal, transforms)

    def mode(self):
        mode = self.loc
        for transform in self.transforms:
            mode = transform(mode)
        return mode

    def rsample(self, sample_shape=torch.Size()):
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

class Policy(nn.Module):
    def __init__(
        self,
        encoder,
        network,
        action_dim,
        init_final=None,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=10.0,
        tanh_squash_distribution=False,
        fixed_std=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.fixed_std = fixed_std

        self.mean_layer = nn.Linear(network.output_dim, action_dim)
        default_init()(self.mean_layer.weight)
        torch.nn.init.zeros_(self.mean_layer.bias)

        if fixed_std is None:
            self.std_layer = nn.Linear(network.output_dim, action_dim)
            default_init()(self.std_layer.weight)
            torch.nn.init.zeros_(self.std_layer.bias)

    def forward(self, observations: torch.Tensor, temperature: float = 1.0, train: bool = False, 
                non_squash_distribution: bool = False) -> D.Distribution:
        if self.encoder is not None:
            with torch.set_grad_enabled(train):
                obs_enc = self.encoder(observations)
        else:
            obs_enc = observations

        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        if self.fixed_std is not None:
            stds = torch.full_like(means, self.fixed_std)
        else:
            if self.std_parameterization == "exp":
                log_stds = self.std_layer(outputs)
                stds = torch.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = torch.nn.functional.softplus(self.std_layer(outputs))
            elif self.std_parameterization == "uniform":
                log_stds = self.log_stds
                stds = torch.exp(log_stds)
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")

        # Clip stds and scale with temperature
        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(torch.tensor(temperature))

        if self.tanh_squash_distribution and not non_squash_distribution:
            distribution = TanhNormal(loc=means, scale=stds)
        else:
            distribution = Normal(loc=means, scale=stds)

        return distribution

    def get_features(self, observations):
        with torch.no_grad():
            return self.encoder(observations)

def ensemblize(critic_class, num_qs):
    class EnsembleCritic(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.critics = nn.ModuleList([
                critic_class(*args, **kwargs) for _ in range(num_qs)
            ])

        def forward(self, *args, **kwargs):
            return torch.stack([critic(*args, **kwargs) for critic in self.critics], dim=0)

    return EnsembleCritic
