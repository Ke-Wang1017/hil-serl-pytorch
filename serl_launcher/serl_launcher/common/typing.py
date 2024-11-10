from typing import Any, Callable, Dict, Sequence, Union

import torch
import numpy as np

# Type for device placement
Device = Union[str, torch.device]
# Type for model parameters
Params = Dict[str, Any]
# Type for tensor shapes
Shape = Sequence[int]
# Type for data types
Dtype = torch.dtype
# Type for info dictionaries returned by environments
InfoDict = Dict[str, float]
# Type for array-like data
Array = Union[np.ndarray, torch.Tensor]
# Type for nested data structures (e.g. observations)
Data = Union[Array, Dict[str, "Data"]]
# Type for batches of data
Batch = Dict[str, Data]
# Type for module methods
ModuleMethod = Union[str, Callable, None]
# Type for random number generators
Generator = torch.Generator
