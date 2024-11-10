from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass, field


def default_init():
    return nn.init.xavier_uniform_


def shard_batch(batch, num_devices):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A dictionary of tensors.
        num_devices: Number of devices to shard across.
    """
    def shard_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.chunk(num_devices, dim=0)
        return x
    
    return {k: shard_tensor(v) for k, v in batch.items()}


class ModuleDict(nn.Module):
    """
    Utility class for wrapping a dictionary of modules. This is useful when you have multiple modules that you want to
    initialize all at once (creating a single state dict), but you want to be able to call them separately
    later. The modules may have sub-modules nested inside them that share parameters (e.g. an image encoder)
    and PyTorch will automatically handle this without duplicating the parameters.

    To initialize the modules, call forward with no name arg, and then pass the example arguments to each module as
    additional kwargs. To call the modules, pass the name of the module as the name arg, and then pass the arguments
    to the module as additional args or kwargs.

    Example usage:
    ```
    shared_encoder = Encoder()
    actor = Actor(encoder=shared_encoder)
    critic = Critic(encoder=shared_encoder)

    model_def = ModuleDict({"actor": actor, "critic": critic})
    
    actor_output = model_def(example_obs, name="actor")
    critic_output = model_def(example_obs, action=example_action, name="critic")
    ```
    """

    def __init__(self, modules: Dict[str, nn.Module]):
        super().__init__()
        self.modules_dict = nn.ModuleDict(modules)

    def forward(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules_dict.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. "
                    f"Got kwargs keys {kwargs.keys()} but module keys {self.modules_dict.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules_dict[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules_dict[key](*value)
                else:
                    out[key] = self.modules_dict[key](value)
            return out

        return self.modules_dict[name](*args, **kwargs)


@dataclass
class TrainState:
    """
    Custom TrainState class to replace Flax's train state.

    Adds support for holding target params and updating them via polyak
    averaging. Also supports multiple optimizers and proper device handling.

    Attributes:
        step: The current training step.
        model: The main model.
        optimizers: Dictionary of optimizers.
        target_model: Optional target network for algorithms that use it.
        device: The device to use for computations.
    """
    step: int = 0
    model: nn.Module = None
    optimizers: Dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    target_model: Optional[nn.Module] = None
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def state_dict(self):
        return {
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizers': {k: opt.state_dict() for k, opt in self.optimizers.items()},
            'target_model': self.target_model.state_dict() if self.target_model is not None else None,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict['step']
        self.model.load_state_dict(state_dict['model'])
        for k, opt in self.optimizers.items():
            opt.load_state_dict(state_dict['optimizers'][k])
        if self.target_model is not None and state_dict['target_model'] is not None:
            self.target_model.load_state_dict(state_dict['target_model'])

    def target_update(self, tau: float) -> None:
        """
        Performs an update of the target params via polyak averaging. The new
        target params are given by:

            new_target_params = tau * params + (1 - tau) * target_params
        """
        if self.target_model is None:
            return

        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def apply_loss_fn(self, loss_fn: Callable, optimizer_key: str) -> Tuple[Any, Any]:
        """
        Applies a loss function and updates parameters using the specified optimizer.
        
        Args:
            loss_fn: Function that computes loss and returns (loss, info)
            optimizer_key: Key of the optimizer to use
            
        Returns:
            Tuple of (loss, info)
        """
        optimizer = self.optimizers[optimizer_key]
        optimizer.zero_grad()
        
        loss, info = loss_fn()
        loss.backward()
        optimizer.step()
        
        self.step += 1
        return loss.item(), info

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        target_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> "TrainState":
        """Creates a new train state."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model = model.to(device)
        if target_model is not None:
            target_model = target_model.to(device)
            target_model.load_state_dict(model.state_dict())
            
        return cls(
            model=model,
            optimizers=optimizers,
            target_model=target_model,
            device=device,
        )
