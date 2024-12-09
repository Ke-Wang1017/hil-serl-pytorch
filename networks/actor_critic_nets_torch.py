from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from networks.mlp_torch import MLP, default_init

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
        init_final: Optional[float] = None,
        activate_final: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        self.activate_final = activate_final
        
        # Output layer
        if init_final is not None:
            if self.activate_final:
                self.output_layer = nn.Linear(network.net[-3].out_features, 1)
            else:
                self.output_layer = nn.Linear(network.net[-2].out_features, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            if self.activate_final:
                self.output_layer = nn.Linear(network.net[-3].out_features, 1)
            else:
                self.output_layer = nn.Linear(network.net[-2].out_features, 1)
            default_init()(self.output_layer.weight)
        
        self.to(self.device)

    def forward(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor,
        train: bool = False
    ) -> torch.Tensor:
        self.train(train)
        
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        
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
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        
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
        output_dim: int = 3,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        self.output_dim = output_dim
        
        # Output layer
        if init_final is not None:
            if self.activate_final:
                self.output_layer = nn.Linear(network.net[-3].out_features, output_dim)
            else:
                self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            if self.activate_final:
                self.output_layer = nn.Linear(network.net[-3].out_features, output_dim)
            else:
                self.output_layer = nn.Linear(network.net[-2].out_features, output_dim)
            default_init()(self.output_layer.weight)
            
        self.to(self.device)
            
    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        self.train(train)
        
        observations = observations.to(self.device)
        
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
        fixed_std: Optional[torch.Tensor] = None,
        init_final: Optional[float] = None,
        activate_final: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.fixed_std = fixed_std.to(self.device) if fixed_std is not None else None
        self.activate_final = activate_final
        
        # Mean layer
        if self.activate_final:
            self.mean_layer = nn.Linear(network.net[-3].out_features, action_dim)
        else:
            self.mean_layer = nn.Linear(network.net[-2].out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            default_init()(self.mean_layer.weight)
        
        # Standard deviation layer or parameter
        if fixed_std is None:
            if std_parameterization == "uniform":
                self.log_stds = nn.Parameter(torch.zeros(action_dim, device=self.device))
            else:
                if self.activate_final:
                    self.std_layer = nn.Linear(network.net[-3].out_features, action_dim)
                else:
                    self.std_layer = nn.Linear(network.net[-2].out_features, action_dim)
                if init_final is not None:
                    nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                    nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
                else:
                    default_init()(self.std_layer.weight)
        
        self.to(self.device)

    def forward(
        self, 
        observations: torch.Tensor,
        temperature: float = 1.0,
        train: bool = False,
        non_squash_distribution: bool = False
    ) -> torch.distributions.Distribution:
        self.train(train)
                
        # Encode observations if encoder exists
        if self.encoder is not None:
            with torch.set_grad_enabled(train):
                obs_enc = self.encoder(observations, train=train)
        else:
            obs_enc = observations
        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)
        
        # Compute standard deviations
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = self.std_layer(outputs)
                stds = torch.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = torch.nn.functional.softplus(self.std_layer(outputs))
            elif self.std_parameterization == "uniform":
                stds = torch.exp(self.log_stds).expand_as(means)
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            assert self.std_parameterization == "fixed"
            stds = self.fixed_std.expand_as(means)

        # Clip standard deviations and scale with temperature
        temperature = torch.tensor(temperature, device=self.device)
        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(temperature)

        # Create distribution
        if self.tanh_squash_distribution and not non_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = torch.distributions.Normal(
                loc=means,
                scale=stds,
            )

        return distribution
    
    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        observations = observations.to(self.device)
        if self.encoder is not None:
            with torch.no_grad():
                return self.encoder(observations, train=False)
        return observations


def create_critic_ensemble(critic_class, num_critics: int, device: str = "cuda") -> nn.ModuleList:
    """Creates an ensemble of critic networks"""
    critics = nn.ModuleList([critic_class() for _ in range(num_critics)])
    return critics.to(device)


class TanhMultivariateNormalDiag(torch.distributions.TransformedDistribution):
    def __init__(
        self,
        loc: torch.Tensor,
        scale_diag: torch.Tensor,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ):
        # Create base normal distribution
        base_distribution = torch.distributions.Normal(loc=loc, scale=scale_diag)
        
        # Create list of transforms
        transforms = []
        
        # Add tanh transform
        transforms.append(torch.distributions.transforms.TanhTransform())
        
        # Add rescaling transform if bounds are provided
        if low is not None and high is not None:
            transforms.append(
                torch.distributions.transforms.AffineTransform(
                    loc=(high + low) / 2,
                    scale=(high - low) / 2
                )
            )
        
        # Initialize parent class
        super().__init__(
            base_distribution=base_distribution,
            transforms=transforms
        )
        
        # Store parameters
        self.loc = loc
        self.scale_diag = scale_diag
        self.low = low
        self.high = high

    def mode(self) -> torch.Tensor:
        """Get the mode of the transformed distribution"""
        # The mode of a normal distribution is its mean
        mode = self.loc
        
        # Apply transforms
        for transform in self.transforms:
            mode = transform(mode)
        
        return mode

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Reparameterized sample from the distribution
        """
        # Sample from base distribution
        x = self.base_dist.rsample(sample_shape)
        
        # Apply transforms
        for transform in self.transforms:
            x = transform(x)
            
        return x

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of a value
        Includes the log det jacobian for the transforms
        """
        # Initialize log prob
        log_prob = torch.zeros_like(value[..., 0])
        
        # Inverse transforms to get back to normal distribution
        q = value
        for transform in reversed(self.transforms):
            q = transform.inv(q)
            log_prob = log_prob - transform.log_abs_det_jacobian(q, transform(q))
        
        # Add base distribution log prob
        log_prob = log_prob + self.base_dist.log_prob(q).sum(-1)
        
        return log_prob

    def sample_and_log_prob(self, sample_shape=torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the distribution and compute log probability
        """
        x = self.rsample(sample_shape)
        log_prob = self.log_prob(x)
        return x, log_prob

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of the distribution
        """
        # Start with base distribution entropy
        entropy = self.base_dist.entropy().sum(-1)
        
        # Add log det jacobian for each transform
        x = self.rsample()
        for transform in self.transforms:
            entropy = entropy + transform.log_abs_det_jacobian(x, transform(x))
            x = transform(x)
            
        return entropy
