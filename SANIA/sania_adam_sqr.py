import torch
from torch import optim

from torch import Tensor
from typing import Iterable, Dict, Any, Tuple, List

class SANIA_AdamSQR(optim.Optimizer):
    
    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]], 
                 lr: float = 1.0,
                 is_sqrt: bool = False, 
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 0.1) -> None: 
        
        defaults = dict(
            is_sqrt=is_sqrt,
            eps=eps,
            betas=betas,
            lr=lr,
            )

        super().__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"] = torch.zeros_like(p)
                
        self._step_t: int = 0

        
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            
            is_sqrt: bool = group["is_sqrt"]
            eps: float = group["eps"]
            betas: Tuple[float, float] = group["betas"]
            lr: float = group["lr"]
            
            # Sum of squared gradients 
            self._step_t += 1
            for p in group["params"]:
                state = self.state[p]
                state["sum"].mul_(betas[1]).add_((1 - betas[1]) * p.grad.data.square())
                
            # Preconditioned gradient (search direction), i.e. inv(B_t)*grad
            preconditioned_grads: List[Tensor] = []
            for p in group["params"]:
                state = self.state[p]
                v_hat = state["sum"] / (1 - betas[1]**self._step_t)
                sum_regularized = torch.maximum(torch.full_like(v_hat, fill_value=eps), v_hat)
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
                det: float = 1 - 2 * c
                if det > 0.0:
                    step_size = (1 - torch.sqrt(det)).item()

             # Update parameters
            with torch.no_grad():
                for s, p in zip(preconditioned_grads, group["params"]):
                    p.sub_(s, alpha=lr * step_size)