from typing import Optional, Union, Tuple, Callable
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def make_optimizer(
    learning_rate: float = 3e-4,
    warmup_steps: int = 0,
    cosine_decay_steps: Optional[int] = None,
    weight_decay: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    return_lr_schedule: bool = False,
) -> Union[optim.Optimizer, Tuple[optim.Optimizer, LambdaLR]]:
    """
    Create an optimizer with learning rate scheduling.
    
    Args:
        learning_rate: Base learning rate
        warmup_steps: Number of warmup steps
        cosine_decay_steps: Number of steps for cosine decay (None for constant LR)
        weight_decay: Weight decay coefficient
        clip_grad_norm: Maximum norm for gradient clipping
        return_lr_schedule: Whether to return the learning rate scheduler
        
    Returns:
        Optimizer or tuple of (optimizer, scheduler)
    """
    def get_lr_lambda(step: int) -> float:
        """Learning rate schedule function."""
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        
        if cosine_decay_steps is not None:
            # Cosine decay after warmup
            step = min(step - warmup_steps, cosine_decay_steps)
            decay_ratio = float(step) / float(cosine_decay_steps)
            coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * 3.14159)))
            return coeff.item()
        
        # Constant learning rate after warmup if no decay
        return 1.0

    # Create optimizer
    def create_optimizer(params) -> optim.Optimizer:
        if weight_decay is not None:
            optimizer = optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.Adam(
                params,
                lr=learning_rate
            )
        return optimizer

    # Create optimizer wrapper with gradient clipping if needed
    if clip_grad_norm is not None:
        class OptimizerWithClipping:
            def __init__(self, params):
                self.optimizer = create_optimizer(params)
                
            def zero_grad(self):
                self.optimizer.zero_grad()
                
            def step(self):
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'],
                    clip_grad_norm
                )
                self.optimizer.step()
                
            def state_dict(self):
                return self.optimizer.state_dict()
            
            def load_state_dict(self, state_dict):
                self.optimizer.load_state_dict(state_dict)
            
            @property
            def param_groups(self):
                return self.optimizer.param_groups
        
        optimizer_class = OptimizerWithClipping
    else:
        optimizer_class = create_optimizer

    # Return optimizer factory function
    def optimizer_fn(params):
        optimizer = optimizer_class(params)
        scheduler = LambdaLR(
            optimizer if not isinstance(optimizer, OptimizerWithClipping) else optimizer.optimizer,
            lr_lambda=get_lr_lambda
        )
        
        if return_lr_schedule:
            return optimizer, scheduler
        return optimizer

    return optimizer_fn
