import torch
from torch import optim

from torch import Tensor
from typing import Any, Dict, Iterable, List

class SANIA_AdagradSQR(optim.Optimizer):
    
    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]], 
                 lr: float = 1.0,
                 is_sqrt: bool = False, 
                 eps: float = 0.1) -> None:
        
        defaults = dict(
            is_sqrt=is_sqrt,
            eps=eps,
            lr=lr,
            )

        super().__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"] = torch.zeros_like(p)

        
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            
            is_sqrt: bool = group["is_sqrt"]
            eps: float = group["eps"]
            lr: float = group["lr"]
            
            # Sum of squared gradients 
            for p in group["params"]:
                state = self.state[p]
                state["sum"].addcmul_(p.grad.data, p.grad.data, value=1)
                
            # Preconditioned gradient (search direction), i.e. inv(B_t)*grad
            preconditioned_grads: List[Tensor] = []
            for p in group["params"]:
                state = self.state[p]
                sum_regularized = torch.maximum(torch.full_like(state["sum"], fill_value=eps), state["sum"])
                if is_sqrt:
                    t = p.grad / torch.sqrt(sum_regularized)
                else:
                    t = p.grad / sum_regularized
                preconditioned_grads.append(t)
                
            # Scaled norm of gradients, i.e. \|g\|^2_{inv(B_t)} == <inv(B_t)g, g>
            grad_norm: float = 0.0
            for s, p in zip(preconditioned_grads, group["params"]):
                grad_norm += s.mul(p.grad.data).sum()

            # Polyak step size
            step_size: float = 1.0 
            if 2 * loss.item() <= grad_norm:
                c = loss.item() / grad_norm
                det = 1 - 2 * c
                if det > 0.0:
                    step_size = (1 - torch.sqrt(det)).item()

            group["lr"] = step_size
            # Update parameters
            with torch.no_grad():
                for s, p in zip(preconditioned_grads, group["params"]):
                    p.sub_(s, alpha=step_size)