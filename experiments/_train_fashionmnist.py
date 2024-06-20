import os
import sys
import argparse
from typing import Tuple, Literal

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import torch.utils.data as data_utils
import torch.optim as optim

import torchvision
import torchvision.models as models

from sklearn.datasets import load_svmlight_file

import matplotlib.pyplot as plt

import experiments.loss_functions as lf
from experiments.utils import load_results, save_results
from experiments.datasets import get_MNIST, get_FashionMNIST

from dotenv import load_dotenv
load_dotenv()

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_model(model, criterion, test_loader, test_hist) -> None:
    test_epoch_loss = 0.0
    total = 0
    correct = 0
    for i, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_target)
        test_epoch_loss += loss.item() * batch_data.size(0)
        
        
        _, predicted = torch.max(outputs.data, 1)
        total += batch_target.size(0)
        batch_correct = (predicted == batch_target).sum().item()
        batch_accuracy = batch_correct / batch_target.size(0)
        correct += batch_correct
        
        test_hist["batch_loss"].append(loss.item())
        test_hist["batch_accuracy"].append(batch_accuracy)

    test_hist["epoch_loss"].append(test_epoch_loss / len(test_loader.sampler))
    test_hist["epoch_accuracy"].append(correct / total)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y
    
class LeNet5X(nn.Module):
    def __init__(self):
        super(LeNet5X, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, 84)
        self.relu5 = nn.ReLU()
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        y = self.fc4(y)
        return y



def run_optimizer(opt, lr, model, criterion, train_loader, test_loader, epochs, seed=0):

    torch.manual_seed(seed)

    optimizer = opt(model.parameters(), lr=lr)
    
    train_hist = {
        "epoch_loss": [],
        "batch_loss": [],
    }
    
    test_hist = {
        "epoch_accuracy": [],
        "batch_accuracy": [],
        "epoch_loss": [],
        "batch_loss": [],
    }
    

    for epoch in range(epochs):
        
        print(f"Epoch: [{epoch}]")
        
        model.eval()
        with torch.inference_mode():
            eval_model(model, criterion, test_loader, test_hist)
            print(f"Test accuracy: {test_hist['epoch_accuracy'][-1]}")
        
        
        train_epoch_loss = 0.0
        model.train()
        for i, (batch_data, batch_target) in enumerate(train_loader): 
            
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()

            outputs = model(batch_data)
            loss = criterion(outputs, batch_target)
            train_epoch_loss += loss.item() * batch_data.size(0)

            loss.backward()
            optimizer.step()
            
            train_hist["batch_loss"].append(loss.item())
            
        train_epoch_loss = train_epoch_loss / len(train_loader.sampler)
        train_hist["epoch_loss"].append(train_epoch_loss)
        print(f"Train loss: {train_epoch_loss:.4f}")

    return {
        "train_hist": train_hist,
        "test_hist": test_hist,
        "model_state_dict": model.state_dict()
    }


def sania_adagrad(precond_method: str, model: nn.Module, criterion: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
                  epochs: int, eps: float = 0.5, seed: int = 0) -> dict:
    
    torch.manual_seed(seed)

    train_hist = {  
        "epoch_loss": [],
        "batch_loss": [],
    }
    test_hist = {
        "epoch_accuracy": [],
        "batch_accuracy": [],
        "epoch_loss": [],
        "batch_loss": [],
    }

    grad_sums = [torch.zeros_like(p) for p in model.parameters()]
    s = [torch.zeros_like(p) for p in model.parameters()]
    
    assert precond_method in ("adagrad_sqr", "adagrad")
    
    eps = torch.tensor(eps)

    for epoch in range(epochs):
        
        print(f"Epoch: [{epoch}]")
                
        model.eval()
        with torch.inference_mode():
            eval_model(model, criterion, test_loader, test_hist)
            print(f"Test accuracy: {test_hist['epoch_accuracy'][-1]}")
            
        train_epoch_loss = 0.0
        model.train()
        for i, (batch_data, batch_target) in enumerate(train_loader): 
            
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            for p in model.parameters():
                p.grad = None
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_target)
            train_epoch_loss += loss.item() * batch_data.size(0)
            
            loss.backward()
            
            # Sum of squared gradients 
            for v, p in zip(grad_sums, model.parameters()):
                v.add_(torch.square(p.grad))
                
            # Gradient direction preconditioned by Adagrad 
            preconditioned_grads = []
            for v, p in zip(grad_sums, model.parameters()):
                grad_sum_regularized = torch.maximum(torch.ones_like(v) * eps, v)

                if precond_method == "adagrad":
                    t = p.grad / torch.sqrt(grad_sum_regularized)
                elif precond_method == "adagrad_sqr":
                    t = p.grad / grad_sum_regularized
                preconditioned_grads.append(t)

            
            # Scaled norm of gradients ||g||_{inv(B_t)}
            grad_norm_sq_scaled = torch.tensor(0.0).to(device)
            for s, p in zip(preconditioned_grads, model.parameters()):
                grad_norm_sq_scaled.add_(s.mul(p.grad).sum())
                

            # Polyak step size
            step_size = 1.0 
            if 2 * loss.item() <= grad_norm_sq_scaled:
                c = loss.item() / ( grad_norm_sq_scaled )
                det = 1 - 2 * c
                if det > 0.0:
                    step_size = (1 - torch.sqrt(det)).item()
                    
            # optimization step
            with torch.no_grad():
                for s, p in zip(preconditioned_grads, model.parameters()):
                    p.sub_(s, alpha=step_size)
                    
            train_hist["batch_loss"].append(loss.item())
            
        train_epoch_loss = train_epoch_loss / len(train_loader.sampler)
        train_hist["epoch_loss"].append(train_epoch_loss)
        print(f"Train loss: {train_epoch_loss:.4f}")
                    
    return {
        "train_hist": train_hist,
        "test_hist": test_hist,
        "model_state_dict": model.state_dict()
    }


def sania_adam(precond_method: Literal["adam", "adam_sqr"], model: nn.Module, 
               criterion: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
               epochs: int, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 0.5, seed: int = 0) -> dict:
    
    torch.manual_seed(seed)

    train_hist = {  
        "epoch_loss": [],
        "batch_loss": [],
    }
    test_hist = {
        "epoch_accuracy": [],
        "batch_accuracy": [],
        "epoch_loss": [],
        "batch_loss": [],
    }

    grad_sums = [torch.zeros_like(p) for p in model.parameters()]
    s = [torch.zeros_like(p) for p in model.parameters()]

    assert precond_method in ("adam_sqr", "adam"), f"{precond_method} is not in (adam_sqr, adam)."
    
    eps = torch.tensor(eps)
    step_t = torch.tensor(0.)

    for epoch in range(epochs):
        
        print(f"Epoch: [{epoch}]")

        model.eval()
        with torch.inference_mode():
            eval_model(model, criterion, test_loader, test_hist)
            print(f"Test accuracy: {test_hist['epoch_accuracy'][-1]}")
        
        train_epoch_loss = 0.0
        model.train()
        for i, (batch_data, batch_target) in enumerate(train_loader): 
            
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            for p in model.parameters():
                p.grad = None
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_target)
            loss.backward()
            train_epoch_loss += loss.item() * batch_data.size(0)
            
            # Sum of squared gradients 
            step_t += 1
            for p, g in zip(model.parameters(), grad_sums):
                g.mul_(betas[1]).add_((1 - betas[1]) * p.grad.data.square())
                
            # Gradient direction preconditioned by Adagrad 
            preconditioned_grads = []
            for v, p in zip(grad_sums, model.parameters()):
                v_hat = v / (1 - torch.pow(betas[1], step_t))
                grad_sum_regularized = torch.maximum(torch.ones_like(v_hat) * eps, v_hat)
                if precond_method == "adam":
                    t = p.grad / torch.sqrt(grad_sum_regularized)
                elif precond_method == "adam_sqr":
                    t = p.grad / grad_sum_regularized
                preconditioned_grads.append(t)
                
            # Scaled norm of gradients ||g||_{inv(B_t)}
            grad_norm_sq_scaled = torch.tensor(0.0).to(device)
            for s, p in zip(preconditioned_grads, model.parameters()):
                grad_norm_sq_scaled.add_(s.mul(p.grad).sum())
                

            # Polyak step size
            step_size = 1.0 
            if 2 * loss.item() <= grad_norm_sq_scaled:
                c = loss.item() / ( grad_norm_sq_scaled )
                det = 1 - 2 * c
                if det > 0.0:
                    step_size = (1 - torch.sqrt(det)).item()
                                
            # optimization step
            with torch.no_grad():
                for s, p in zip(preconditioned_grads, model.parameters()):
                    p.sub_(s, alpha=step_size)
            
            train_hist["batch_loss"].append(loss.item())
            
        train_epoch_loss = train_epoch_loss / len(train_loader.sampler)
        train_hist["epoch_loss"].append(train_epoch_loss)
        print(f"Train loss: {train_epoch_loss:.4f}")
        
    return {
        "train_hist": train_hist,
        "test_hist": test_hist,
        "model_state_dict": model.state_dict()
    }





def main(model_name: str, optimizer: str, epochs: int, train_batch_size: int, test_batch_size: int, 
         lr: float = 1.0, scale: int = 0, save: bool = False, seed: int = 0) -> None:
    
    torch.manual_seed(seed)

    if model_name in ("LeNet5", "lenet5"):
        model = LeNet5().to(device)
    elif model_name in ("LeNet5X", "lenet5x"):
        model = LeNet5X().to(device)

    criterion = nn.CrossEntropyLoss()
    
    train_loader, test_loader = get_FashionMNIST(train_batch_size, test_batch_size, seed=0)
    
    if optimizer in ("adam", "Adam"):
        result = run_optimizer(optim.Adam, lr, model, criterion, train_loader, test_loader, epochs, seed=seed)
    elif optimizer in ("adagrad", "Adagrad", "AdaGrad"):
        result = run_optimizer(optim.Adagrad, lr, model, criterion, train_loader, test_loader, epochs, seed=seed)
    elif optimizer == "sania_adam":
        result = sania_adam("adam", model, criterion, train_loader, test_loader, epochs, seed=seed)
    elif optimizer == "sania_adam_sqr":
        result = sania_adam("adam_sqr", model, criterion, train_loader, test_loader, epochs, seed=seed)        
    elif optimizer == "sania_adagrad":
        result = sania_adagrad("adagrad", model, criterion, train_loader, test_loader, epochs, seed=seed)
    elif optimizer == "sania_adagrad_sqr":
        result = sania_adagrad("adagrad_sqr", model, criterion, train_loader, test_loader, epochs, seed=seed)

    if save:
        save_results(result, "FashionMNIST", scale, train_batch_size, epochs, optimizer, lr, seed=seed)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--model", type=str, choices=["lenet5", "lenet5x", "LeNet5", "LeNet5X"])
    parser.add_argument("--optimizer", type=str, choices=["adam", "adagrad", "sania_adam", "sania_adam_sqr", "sania_adagrad", "sania_adagrad_sqr"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=0, help="Scaling range. [-scale, scale].")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run, i.e. --seeds=3 will run on seeds [0, 1, 2]")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    args = parser.parse_args()
    print(f"device: {device}")
    print(args)
    
    import time 
    import datetime

    start_time = time.time()

    for seed in range(args.seeds):
        print(f"Seed: {seed}")
        main(model_name=args.model, optimizer=args.optimizer, epochs=args.epochs, 
            train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            lr=args.lr, scale=args.scale, save=args.save, seed=seed)
        
    c = round(time.time() - start_time, 2)
    print(f"Run complete in {str(datetime.timedelta(seconds=c))} hrs:min:sec.")
    