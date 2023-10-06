import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.optim import SGD, Adam, Adagrad, Adadelta, RMSprop
from torch_optimizer import Adahessian

from dotenv import load_dotenv
load_dotenv()

import datasets
import models
from projects.ScaledSPS.loss_functions import get_loss
from utils import restricted_float

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_results(result, dataset, percent, scale, batch_size, epochs, loss_class, optimizer_class, lr, preconditioner, slack_method, lmd, mu, seed):
    results_path = os.getenv("RESULTS_DIR")
    directory = f"{results_path}/{dataset}/percent_{percent}/scale_{scale}/bs_{batch_size}" \
    f"/epochs_{epochs}/{loss_class}/{optimizer_class}/lr_{lr}/precond_{preconditioner}/slack_{slack_method}/lmd_{lmd}/mu_{mu}/seed_{seed}"
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    torch.save([x[0] for x in result], f"{directory}/loss")
    torch.save([x[1] for x in result], f"{directory}/grad_norm_sq")
    
    if optimizer_class in ("sps", "sps2"):
        torch.save([x[2] for x in result], f"{directory}/slack")



@torch.no_grad()
def eval_model(model, loss_fn, data_loader):
    n_correct = 0
    n_samples = 0
    loss = 0
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss += loss_fn(outputs, labels).item() / len(data_loader)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()  
    
    acc = 100.0 * n_correct / n_samples

    return loss, acc


def run_optimizer(model, criterion, train_loader, test_loader, epochs, log_every_n_epochs, optimizer_class, **optimizer_kwargs):
    
    torch.manual_seed(0)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    hist = []

        
    for epoch in range(epochs):

        if epoch % log_every_n_epochs == 0:
            with torch.no_grad():
                train_loss, train_acc = eval_model(model, criterion, train_loader) 
                print(f"[{epoch}] | Train Loss: {train_loss} | Train Acc: {train_acc}")
                test_loss, test_acc = eval_model(model, criterion, test_loader)
                print(f"[{epoch}] | Test Loss: {test_loss} | Test Acc: {test_acc}")
                hist.append([train_loss, train_acc, test_loss, test_acc])

        for i, (images, labels) in enumerate(train_loader):  
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            def closure():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward(create_graph=True)
                return loss
            
            if isinstance(optimizer, Adahessian): 
                optimizer.step(closure) 
            else:
                loss = closure()
                optimizer.step()
        
    return hist


models_dict = {
    "NN": models.NN,
    "LeNet": models.LeNet
}

criterions_dict = {
    "cross-entropy": torch.nn.CrossEntropyLoss(),
}

datasets_dict = {
    "mnist": datasets.get_MNIST,
    "MNIST": datasets.get_MNIST
}

optimizers_dict = {
    "adam": Adam,
    "adagrad": Adagrad,
    "adahessian": Adahessian,
}


def run(model_name, criterion_name, dataset_name, 
        batch_size, percentage, scale, epochs, optimizer_name, learning_rate, loss_name, seed):

    torch.random.manual_seed(seed)

    model = models_dict.get(model_name)
    criterion = criterions_dict.get(criterion_name)
    train_loader, test_loader = datasets_dict.get(dataset_name)(batch_size, percentage)
    optimizer = optimizers_dict.get(optimizer_name)

    hist = run_optimizer(model, criterion, train_loader, test_loader, epochs, optimizer, learning_rate=learning_rate)

    save_results(result=hist, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="sgd", lr=round(lr, 5), preconditioner="none", slack_method="none", lmd=lmd, mu=mu, 
                seed=seed)



def main(lr, loss_name):
    scales = [0, 3, 5]
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        for scale in scales:
            for dataset_name, batch_size, percentage in datasets_setup:
                print(dataset_name, batch_size, percentage, scale, lr, seed)
                run(dataset_name=dataset_name, batch_size=batch_size, percentage=percentage, scale=scale, epochs=500, learning_rate=lr, loss_name=loss_name, seed=seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--lr", type=float, default=-1.0, help="-1.0 to choose optimized hyperparameter from optuna results.")
    parser.add_argument("--loss", type=str, choices=["logreg", "nllsq"])

    args = parser.parse_args()
    print(f"device: {device}")
    main(lr=args.lr, loss_name=args.loss)