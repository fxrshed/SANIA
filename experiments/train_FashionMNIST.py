import os
import sys
import argparse
import time 
import datetime
from collections import defaultdict

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2
import torch.optim as optim

from utils import get_FashionMNIST, save_results

from SANIA import SANIA_AdamSQR, SANIA_AdagradSQR

import neptune

from dotenv import load_dotenv
load_dotenv()

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2) # COMMENT OUT IF CPU IS NOT LIMITED
os.environ["OMP_NUM_THREADS"] = "1" # !!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TORCHVISION_DATASETS_DIR = os.getenv("TORCHVISION_DATASETS_DIR")

@torch.inference_mode
def eval_model(model: nn.Module, criterion: nn.Module, test_loader: DataLoader) -> tuple[float, float, list[float], list[float]]:
    
    test_epoch_loss = 0.0
    
    test_batch_loss = []
    test_batch_acc = []
    
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
        correct += batch_correct
        
        test_batch_loss.append(loss.item())
        test_batch_acc.append(batch_correct)
    
    test_epoch_loss = test_epoch_loss / len(test_loader.sampler)
    test_epoch_acc = correct / total
    return test_epoch_loss, test_epoch_acc, test_batch_loss, test_batch_acc

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
    
class SimpleConvNet(nn.Module):
    
    # https://github.com/kefth/fashion-mnist/blob/master/model.py

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x

def run_optimizer(model: nn.Module, 
                  optimizer: nn.Module, 
                  train_loader: DataLoader, test_loader: DataLoader, 
                  n_epochs: int, 
                  seed: int = 0,
                  neptune_mode: str = "async") -> dict:

    torch.manual_seed(seed)
    
    criterion = nn.CrossEntropyLoss()
    
    history = defaultdict(list)
    
    run = neptune.init_run(
        mode=neptune_mode,
        tags=["binary-classification", "DNN"],
    )
    
    run["dataset"] = {
        "dataset_name": "FashionMNIST",
        "train_batch_size": train_loader.batch_size,
        "test_batch_size": test_loader.batch_size, 
    }
    run["n_epochs"] = n_epochs
    run["seed"] = seed
    
    run["optimizer/parameters/name"] = optimizer.__class__.__name__
    run["optimizer/parameters"] = neptune.utils.stringify_unsupported(optimizer.defaults)
    
    run["model"] = model._get_name()
    run["device"] = str(device)
    
    for epoch in range(n_epochs):

        model.eval()
        with torch.inference_mode():
            test_loss, test_acc, test_batch_loss, test_batch_acc = eval_model(model=model, criterion=criterion, test_loader=test_loader)
            
            history["test/epoch/loss"].append(test_loss)
            history["test/epoch/acc"].append(test_acc)
            history["test/batch/loss"] += test_batch_loss
            history["test/batch/acc"] += test_batch_acc
            
            run["test/loss"].append(test_loss)
            run["test/acc"].append(test_acc)

        print(f"Epoch [{epoch}/{n_epochs}] | Test Loss: {test_loss} | Test Acc: {test_acc}")
        
        train_loss = 0.0
        total = 0
        correct = 0

        model.train()
        for i, (batch_data, batch_target) in enumerate(train_loader): 
            
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
        
            closure = lambda: criterion(outputs, batch_target)
            outputs = model(batch_data)
            loss = closure()
            train_loss += loss.item() * batch_data.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_target.size(0)
            batch_correct = (predicted == batch_target).sum().item()
            correct += batch_correct

            loss.backward()
            optimizer.step(closure=closure)
            
            history["train/batch/loss"].append(loss.item())
            history["train/batch/acc"].append(batch_correct)

            
        history["train/epoch/loss"].append(train_loss / len(train_loader.sampler))
        history["train/epoch/acc"].append(correct / total)
        
        run["train/loss"].append(train_loss / len(train_loader.sampler))
        run["train/acc"].append(correct / total)
        
    run.stop()

    return history


model_dict = {
    "SimpleConvNet": SimpleConvNet,
    "LeNet5X": LeNet5X,
}

optim_dict = {
    "Adam": optim.Adam,
    "Adagrad": optim.Adagrad,
    "SANIA_AdamSQR": SANIA_AdamSQR,
    "SANIA_AdagradSQR": SANIA_AdagradSQR,
}

def main(model_name: str, optimizer_name:str, lr: float, eps: float, n_epochs: int, train_batch_size: int, test_batch_size: int, 
         save: bool = True, seed: int = 0, neptune_mode: str = "async") -> None:
    
    torch.manual_seed(0)

    model = model_dict[model_name]().to(device)
    if optimizer_name in ["Adam", "Adagrad"]:
        optimizer = optim_dict[optimizer_name](model.parameters(), lr=lr)
    else:
        optimizer = optim_dict[optimizer_name](model.parameters(), lr=lr, eps=eps)
    
    train_transforms = v2.Compose([
        v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.1307,), (0.3081,),
        ),
    ])

    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Normalize(
            (0.1307,), (0.3081,),
        ),
    ])

    train_data = torchvision.datasets.FashionMNIST(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=train_transforms
        )

    test_data = torchvision.datasets.FashionMNIST(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=test_transforms
        )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]
    
    for seed in seeds:
    
        history = run_optimizer(model=model, 
                                optimizer=optimizer,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                n_epochs=n_epochs,
                                seed=seed,
                                neptune_mode=neptune_mode)

        if save:
            save_results(results=history,
                        model_name=model_name,
                        dataset_name="FashionMNIST",
                        scale=0,
                        batch_size=train_batch_size,
                        n_epochs=n_epochs,
                        optimizer=optimizer_name,
                        lr=lr,
                        seed=seed)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--model", type=str, choices=["SimpleConvNet", "LeNet5X"])
    parser.add_argument("--optimizer", type=str, choices=["Adam", "Adagrad", "SANIA_AdamSQR", "SANIA_AdagradSQR"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--neptune_mode", type=str, default="async", choices=["async", "debug", "offline", "read-only", "sync"])
    args = parser.parse_args()
    print(f"device: {device}")
    print(args)
    
    start_time = time.time()

    main(model_name=args.model, optimizer_name=args.optimizer, lr=args.lr, eps=args.eps, n_epochs=args.n_epochs, 
            train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
            save=args.save, seed=args.seed, neptune_mode=args.neptune_mode)
        
    c = round(time.time() - start_time, 2)
    print(f"Run complete in {str(datetime.timedelta(seconds=c))} hrs:min:sec.")