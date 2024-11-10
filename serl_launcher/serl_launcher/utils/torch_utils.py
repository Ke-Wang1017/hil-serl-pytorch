import torch
from typing import Dict, Tuple, Union, Optional, List


def batch_to_device(batch: dict, device: Optional[torch.device] = None) -> dict:
    """Move batch data to specified device."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: _to_device(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(_to_device(v) for v in x)
        return x
    
    return _to_device(batch)


class TorchRNG:
    """A convenient stateful PyTorch RNG wrapper. Can be used to wrap RNG inside
    pure functions.
    """
    
    @classmethod
    def from_seed(cls, seed: int) -> 'TorchRNG':
        generator = torch.Generator()
        generator.manual_seed(seed)
        return cls(generator)

    def __init__(self, generator: torch.Generator):
        self.generator = generator

    def __call__(self, keys: Optional[Union[int, List[str]]] = None) -> Union[torch.Generator, Tuple[torch.Generator, ...], Dict[str, torch.Generator]]:
        if keys is None:
            # Create a new generator with different seed
            new_gen = torch.Generator()
            new_gen.manual_seed(torch.randint(high=2**32, size=(1,), generator=self.generator).item())
            return new_gen
        elif isinstance(keys, int):
            # Create multiple generators
            generators = []
            for _ in range(keys):
                new_gen = torch.Generator()
                new_gen.manual_seed(torch.randint(high=2**32, size=(1,), generator=self.generator).item())
                generators.append(new_gen)
            return tuple(generators)
        else:
            # Create dictionary of generators
            generators = {}
            for key in keys:
                new_gen = torch.Generator()
                new_gen.manual_seed(torch.randint(high=2**32, size=(1,), generator=self.generator).item())
                generators[key] = new_gen
            return generators


def wrap_function_with_rng(generator: torch.Generator):
    """To be used as decorator, automatically bookkeep a RNG for the wrapped function."""
    
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal generator
            # Create new generator with different seed
            new_gen = torch.Generator()
            new_gen.manual_seed(torch.randint(high=2**32, size=(1,), generator=generator).item())
            return function(new_gen, *args, **kwargs)
        return wrapped
    return wrap_function


# Global RNG instance
torch_utils_rng = None


def init_rng(seed: int):
    """Initialize global RNG with seed."""
    global torch_utils_rng
    torch_utils_rng = TorchRNG.from_seed(seed)


def next_rng(*args, **kwargs) -> Union[torch.Generator, Tuple[torch.Generator, ...], Dict[str, torch.Generator]]:
    """Get next RNG state(s) from global RNG."""
    global torch_utils_rng
    return torch_utils_rng(*args, **kwargs) 