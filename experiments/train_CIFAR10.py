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
import torchvision.models as models
import torch.optim as optim

from utils import save_results, evaluate_classification_model

from SANIA import SANIA_AdamSQR, SANIA_AdagradSQR

import neptune

from dotenv import load_dotenv
load_dotenv()

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2) # COMMENT OUT IF CPU IS NOT LIMITED
os.environ["OMP_NUM_THREADS"] = "1" # !!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TORCHVISION_DATASETS_DIR = os.getenv("TORCHVISION_DATASETS_DIR")


def run_optimizer(model: nn.Module, 
                  model_name: str,
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
        "dataset_name": "CIFAR10",
        "train_batch_size": train_loader.batch_size,
        "test_batch_size": test_loader.batch_size, 
    }
    run["n_epochs"] = n_epochs
    run["seed"] = seed
    
    run["optimizer/parameters/name"] = optimizer.__class__.__name__
    run["optimizer/parameters"] = neptune.utils.stringify_unsupported(optimizer.defaults)
   
    run["model"] = model_name
    run["device"] = str(device)
    
    for epoch in range(n_epochs):

        model.eval()
        with torch.inference_mode():
            test_loss, test_acc, test_batch_loss, test_batch_acc = evaluate_classification_model(model=model, criterion=criterion, test_loader=test_loader, device=device)
            
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


models_dict = {
    "ResNet18": models.resnet18,
    "DenseNet121": models.densenet121,
    "WideResNet50": models.wide_resnet50_2,
}

optim_dict = {
    "Adam": optim.Adam,
    "Adagrad": optim.Adagrad,
    "SANIA_AdamSQR": SANIA_AdamSQR,
    "SANIA_AdagradSQR": SANIA_AdagradSQR,
}

def main(model_name: str, optimizer_name: str, lr: float, eps: float, n_epochs: int, train_batch_size: int, test_batch_size: int, 
         save: bool = True, seed: int = 0, neptune_mode: str = "async") -> None:
    
    torch.manual_seed(0)

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(32, 32), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261),
        ),
    ])

    test_transforms = v2.Compose([
        v2.Resize(size=(32, 32)),
        v2.CenterCrop((32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261),
        ),
    ])

    train_data = torchvision.datasets.CIFAR10(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=train_transforms
        )
    test_data = torchvision.datasets.CIFAR10(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=test_transforms
        )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    model = models_dict[model_name](num_classes=len(train_loader.dataset.classes)).to(device)
    if optimizer_name in ["Adam", "Adagrad"]:
        optimizer = optim_dict[optimizer_name](model.parameters(), lr=lr)
    else:
        optimizer = optim_dict[optimizer_name](model.parameters(), lr=lr, eps=eps)

    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]
    
    for seed in seeds:
        
        history = run_optimizer(model=model, 
                                model_name=model_name,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                n_epochs=n_epochs,
                                seed=seed,
                                neptune_mode=neptune_mode)
        
        
        if save:
            save_results(results=history,
                        model_name=model_name,
                        dataset_name="CIFAR10",
                        scale=0,
                        batch_size=train_batch_size,
                        n_epochs=n_epochs,
                        optimizer=optimizer_name,
                        lr=lr,
                        seed=seed)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "Adagrad", "SANIA_AdamSQR", "SANIA_AdagradSQR"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--neptune_mode", type=str, default="async", choices=["async", "debug", "offline", "read-only", "sync"])
    args = parser.parse_args()
    print(f"device: {device}")
    print(args)
    
    start_time = time.time()

    main(model_name="DenseNet121", optimizer_name=args.optimizer, lr=args.lr, eps=args.eps, n_epochs=args.n_epochs, 
        train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
        save=args.save, seed=args.seed, neptune_mode=args.neptune_mode)
        
    c = round(time.time() - start_time, 2)
    print(f"Run completed in {str(datetime.timedelta(seconds=c))} hrs:min:sec.")