from typing import Tuple

import numpy as np

class BaseOptimizer(object):
    
    def step(self, loss=None, grad=None):
        raise NotImplementedError

class SGD(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, lr: float = 1.0):
        self.params = params
        self.lr = lr 
        
        self.defaults = dict(lr=lr)
        
    def step(self, loss, grad):
        
        self.params -= self.lr * grad
            
        return loss, grad
    

class Adagrad(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, lr: float = 0.01, eps: float = 1e-10):
        self.params = params
        self.lr = lr 
        self.eps = eps
        
        self.sum = np.zeros_like(params)
        
        self.defaults = dict(lr=lr, eps=eps)
        
    def step(self, loss, grad):
        self.sum += np.square(grad)
        self.params -= self.lr * (grad / (np.sqrt(self.sum) + self.eps))
            
        return loss, grad
    
    
class Adam(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8):
        self.params = params
        self.lr = lr 
        self.eps = eps
        self.betas = betas
        
        self.sum_m = np.zeros_like(params)
        self.sum_v = np.zeros_like(params)
        self._step_t: int = 0
        
        self.defaults = dict(lr=lr, eps=eps, betas=betas)
        
    def step(self, loss, grad):
        
        self._step_t += 1

        self.sum_m = self.betas[0] * self.sum_m + (1 - self.betas[0]) * grad
        self.sum_v = self.betas[1] * self.sum_v + (1 - self.betas[1]) * np.square(grad)
        m_hat = self.sum_m / (1 - self.betas[0]**self._step_t)
        v_hat = self.sum_v / (1 - self.betas[1]**self._step_t)
    
        self.params -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            
        return loss, grad
    
class SANIA(BaseOptimizer):
    def __init__(self, params: np.ndarray, 
                 lr: float = 1.0):
        self.params = params
        self.lr = lr 
        
        self.defaults = dict(lr=lr)
        
    def step(self, loss: np.ndarray, grad: np.ndarray, preconditioner: np.ndarray):
        
        s = grad / preconditioner
        
        # Scaled norm of gradients ||g||_{inv(B_t)}
        grad_norm = np.dot(grad, s)

        # Polyak step size
        step_size = 1.0 
        if 2 * loss.item() <= grad_norm:
            c = loss.item() / ( grad_norm )
            det = 1 - 2 * c
            if det > 0.0:
                step_size = 1 - np.sqrt(det)
        
        self.params -= self.lr * step_size * s
        
        return loss, grad
    
    
class SANIA_AdagradSQR(BaseOptimizer):
    def __init__(self, params: np.ndarray, 
                 is_sqrt: bool = False,
                 lr: float = 1.0, 
                 eps: float = 1e-8):
        self.params = params
        self.lr = lr 
        self.is_sqrt = is_sqrt
        self.eps = eps
        
        self.sum = np.zeros_like(params)
        
        self.defaults = dict(lr=lr, is_sqrt=is_sqrt, eps=eps)
        
    def step(self, loss, grad):
    
        self.sum += np.square(grad)
        
        if self.is_sqrt:
            s = grad / (np.sqrt(self.sum) + self.eps)
        else:
            s = grad / (self.sum + self.eps)
            
        # Scaled norm of gradients ||g||_{inv(B_t)}
        grad_norm = np.dot(grad, s)

        # Polyak step size
        step_size = 1.0 
        if 2 * loss.item() <= grad_norm:
            c = loss.item() / ( grad_norm )
            det = 1 - 2 * c
            if det > 0.0:
                step_size = 1 - np.sqrt(det)
        
        self.params -= self.lr * step_size * s
        
        return loss, grad

class SANIA_AdamSQR(BaseOptimizer):
    def __init__(self, params: np.ndarray, 
                 is_sqrt: bool = False,
                 lr: float = 1.0, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8):
        self.params = params
        self.lr = lr 
        self.is_sqrt = is_sqrt
        self.eps = eps
        self.betas = betas
        
        self.sum_v = np.zeros_like(params)
        self._step_t: int = 0
        
        self.defaults = dict(lr=lr, is_sqrt=is_sqrt, eps=eps, betas=betas)
        
    def step(self, loss, grad):
        
        self._step_t += 1

        self.sum_v = self.betas[1] * self.sum_v + (1 - self.betas[1]) * np.square(grad)
        v_hat = self.sum_v / (1 - self.betas[1]**self._step_t)
        
        if self.is_sqrt:
            s = grad / (np.sqrt(v_hat) + self.eps)
        else:
            s = grad / (v_hat + self.eps)
            
        # Scaled norm of gradients ||g||_{inv(B_t)}
        
        grad_norm = np.dot(grad, s)

        # Polyak step size
        step_size = 1.0 
        if 2 * loss.item() <= grad_norm:
            c = loss.item() / ( grad_norm )
            det = 1 - 2 * c
            if det > 0.0:
                step_size = 1 - np.sqrt(det)
        
        self.params -= self.lr * step_size * s

        return loss, grad
