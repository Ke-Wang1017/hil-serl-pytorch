from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from mlp_torch import MLP, default_init

class TanhNormal(TransformedDistribution):
    """Represents a distribution of tanh-transformed normal samples."""
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        super().__init__(Normal(loc, scale), [TanhTransform()])
        
    def mode(self) -> torch.Tensor:
        return torch.tanh(self.base_dist.loc)
    
    def sample_and_log_prob(self, sample_shape=torch.Size()):
        samples = self.rsample(sample_shape)
        log_probs = self.log_prob(samples)
        return samples, log_probs

class ValueCritic(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        network: nn.Module,
        init_final: Optional[float] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        
        # Output layer
        if init_final is not None:
            self.output_layer = nn.Linear(network.net[-2].out_features, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.net[-2].out_features, 1)
            default_init()(self.output_layer.weight)
            
    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        self.train(train)
        x = self.network(self.encoder(observations, train))
        value = self.output_layer(x)
        return value.squeeze(-1)

class Critic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        init_final: Optional[float] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        
        # Output layer
        if init_final is not None:
            self.output_layer = nn.Linear(network.net[-2].out_features, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.net[-2].out_features, 1)
            default_init()(self.output_layer.weight)

    def forward(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor,
        train: bool = False
    ) -> torch.Tensor:
        self.train(train)
        
        if self.encoder is not None:
            obs_enc = self.encoder(observations)
        else:
            obs_enc = observations
            
        inputs = torch.cat([obs_enc, actions], dim=-1)
        x = self.network(inputs)
        value = self.output_layer(x)
        return value.squeeze(-1)
    
    def q_value_ensemble(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        train: bool = False
    ) -> torch.Tensor:
        """Forward pass with multiple actions on each state"""
        if len(actions.shape) == 3:  # [batch_size, num_actions, action_dim]
            batch_size, num_actions = actions.shape[:2]
            obs_expanded = observations.unsqueeze(1).expand(-1, num_actions, -1)
            obs_flat = obs_expanded.reshape(-1, observations.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])
            q_values = self(obs_flat, actions_flat, train)
            return q_values.reshape(batch_size, num_actions)
        else:
            return self(observations, actions, train)

class GraspCritic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        init_final: Optional[float] = None,
        output_dim: int = 3
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        self.output_dim = output_dim
        
        # Output layer
        if init_final is not None:
            self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
            default_init()(self.output_layer.weight)
            
    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        self.train(train)
        
        if self.encoder is not None:
            obs_enc = self.encoder(observations)
        else:
            obs_enc = observations
            
        x = self.network(obs_enc)
        return self.output_layer(x)  # [batch_size, output_dim]

class Policy(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        action_dim: int,
        std_parameterization: str = "exp",
        std_min: float = 1e-5,
        std_max: float = 10.0,
        tanh_squash_distribution: bool = False,
        fixed_std: Optional[torch.Tensor] = None
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
        
        # Mean and std layers
        self.mean_layer = nn.Linear(network.net[-2].out_features, action_dim)
        default_init()(self.mean_layer.weight)
        
        if fixed_std is None:
            self.std_layer = nn.Linear(network.net[-2].out_features, action_dim)
            default_init()(self.std_layer.weight)
            
    def forward(
        self, 
        observations: torch.Tensor,
        temperature: float = 1.0,
        train: bool = False,
        non_squash_distribution: bool = False
    ) -> TransformedDistribution:
        self.train(train)
        
        if self.encoder is not None:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)
        else:
            obs_enc = observations
            
        features = self.network(obs_enc)
        means = self.mean_layer(features)
        
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = self.std_layer(features)
                stds = torch.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = nn.functional.softplus(self.std_layer(features))
            elif self.std_parameterization == "uniform":
                log_stds = self.std_layer.bias  # Using bias as learnable parameter
                stds = torch.exp(log_stds).expand_as(means)
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")
        else:
            assert self.std_parameterization == "fixed"
            stds = self.fixed_std.expand_as(means)
            
        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(torch.tensor(temperature))
        
        if self.tanh_squash_distribution and not non_squash_distribution:
            return TanhNormal(means, stds)
        else:
            return Normal(means, stds)
    
    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            with torch.no_grad():
                return self.encoder(observations, train=False)
        return observations

def create_critic_ensemble(critic_class, num_critics: int) -> nn.ModuleList:
    """Creates an ensemble of critic networks"""
    return nn.ModuleList([critic_class() for _ in range(num_critics)]) 