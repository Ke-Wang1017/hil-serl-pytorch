from functools import partial
from typing import Optional, Sequence
import torch
import torch.nn as nn

class LagrangeMultiplier(nn.Module):
    def __init__(
        self,
        init_value: float = 1.0,
        constraint_shape: Sequence[int] = (),
        constraint_type: str = "eq",  # One of ("eq", "leq", "geq")
        parameterization: Optional[str] = None,  # One of ("softplus", "exp"), or None for equality constraints
    ):
        super().__init__()
        self.constraint_type = constraint_type
        self.parameterization = parameterization
        
        # Validate inputs
        if constraint_type != "eq":
            assert init_value > 0, "Inequality constraints must have non-negative initial multiplier values"
            
            if parameterization == "softplus":
                init_value = torch.log(torch.exp(torch.tensor(init_value)) - 1)
            elif parameterization == "exp":
                init_value = torch.log(torch.tensor(init_value))
            elif parameterization == "none":
                pass
            else:
                raise ValueError(f"Invalid multiplier parameterization {parameterization}")
        else:
            assert parameterization is None, "Equality constraints must have no parameterization"
            
        # Initialize the Lagrange multiplier as a parameter
        self.lagrange = nn.Parameter(
            torch.full(constraint_shape, init_value, dtype=torch.float32)
        )

    def forward(
        self, 
        lhs: Optional[torch.Tensor] = None, 
        rhs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get the multiplier value based on parameterization
        multiplier = self.lagrange
        
        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = torch.nn.functional.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = torch.exp(multiplier)
            elif self.parameterization == "none":
                pass
            else:
                raise ValueError(f"Invalid multiplier parameterization {self.parameterization}")
                
        # Return the raw multiplier if no constraint values provided
        if lhs is None:
            return multiplier
            
        # Use the multiplier to compute the Lagrange penalty
        if rhs is None:
            rhs = torch.zeros_like(lhs)
            
        diff = lhs - rhs
        
        assert diff.shape == multiplier.shape, f"Shape mismatch: {diff.shape} vs {multiplier.shape}"
        
        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff
        else:
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")

# Convenience classes with preset configurations
GeqLagrangeMultiplier = partial(
    LagrangeMultiplier, 
    constraint_type="geq", 
    parameterization="softplus"
)

LeqLagrangeMultiplier = partial(
    LagrangeMultiplier, 
    constraint_type="leq", 
    parameterization="softplus"
)

BetterLeqLagrangeMultiplier = partial(
    LagrangeMultiplier, 
    constraint_type="leq", 
    parameterization="none"
)
