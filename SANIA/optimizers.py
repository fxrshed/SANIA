import torch
from torch import optim

from torch import Tensor
from typing import Any, Dict, Iterable, List


class KATE(optim.Optimizer):
    
    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]], 
                 lr: float = 1.0,
                 eps: float = 1e-8,
                 eta: float = 1e-3,
                 ) -> None:
        
        defaults = dict(
            lr=lr,
            eps=eps,
            eta=eta
            )

        super().__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum_b"] = torch.zeros_like(p)
                state["sum_m"] = torch.zeros_like(p)

        
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            
            lr: float = group["lr"]
            eps: float = group["eps"]
            eta: bool = group["eta"]
            
            # Sum of squared gradients 
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data.clone()
                grad_sq = grad * grad
                state["sum_b"].add_(grad_sq)
                state["sum_m"].add_(eta * grad_sq + (grad_sq / (state["sum_b"] + eps)))
                
                # Update parameters
                with torch.no_grad():
                    p.sub_( (torch.sqrt(state["sum_m"]) * p.grad) / (state["sum_b"] + eps), alpha=lr)