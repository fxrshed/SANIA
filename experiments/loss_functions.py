import torch
import numpy as np
class BaseOracle(object):
    
    def func(self, x, data, target):
        raise NotImplementedError
    
    def grad(self, x, data, target):
        raise NotImplementedError
    
    def hess(self, x, data, target):
        raise NotImplementedError

class LogisticRegressionLoss(BaseOracle):
    
    def __init__(self, lmd: float = 0.0) -> None:
        self.lmd = lmd
   
    def func(self, x, data, target):
        return np.mean(np.log(1 + np.exp( - data@x * target ))) + self.lmd/2 * np.linalg.norm(x)**2
    
    def grad(self, x, data, target):
        r = np.exp( - data@x * target )
        ry = -r/(1+r) * target
        return (data.T @ ry )/data.shape[0]  + self.lmd * x
    
    def hess(self, x, data, target):
        r = np.exp( - data@x * target )
        rr= r/(1+r)**2
        return (data.T@np.diagflat(rr)@data) / data.shape[0] + self.lmd*np.eye(data.shape[1])
    
    def func_grad_acc(self, x, data, target):
        sparse_dot = data@x
        t = - sparse_dot * target
        f_val = np.mean(np.log(1 + np.exp( t ))) + self.lmd/2 * np.linalg.norm(x)**2
        
        r = np.exp(t)
        ry = -r/(1+r) * target
        grad_val = (data.T @ ry )/data.shape[0]  + self.lmd * x
        
        acc = (np.sign(sparse_dot) == target).sum() / target.shape[0]

        return f_val, grad_val, acc
        



def logreg(w, X, y):
    return torch.mean(torch.log(1 + torch.exp(-y * (X @ w))))

def grad_logreg(w, X, y):
    r = torch.exp(-y * (X @ w))
    return ( (r/(1 + r)) @ (X * -y[:, None]) ) / X.shape[0]

def hess_logreg(w, X, y):
    r = torch.exp(-y * (X @ w))
    return ( X.T @ (  (r/torch.square(1 + r)).reshape(-1, 1) * X ) ) / X.shape[0]


def nllsq(w, X, y):
    return torch.mean( ( y - (1/(1 + torch.exp(-X @ w ))) )**2 )


def mse(w, X, y):
    return 0.5 * ( torch.sum(torch.square(X @ w - y))  ) / X.shape[0]

def grad_mse(w, X, y):
    return (X.T @ (X @ w - y)) / X.shape[0]

def hess_mse(w, X, y):
    return  (X.T @ X) / X.shape[0]