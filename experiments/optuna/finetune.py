import os
import sys
import urllib.request
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data_utils
from torch.optim import SGD, Adam, Adagrad, Adadelta

from sklearn.datasets import load_svmlight_file

import loss_functions as lf

import optuna
import argparse

from dotenv import load_dotenv
load_dotenv()

torch.set_default_dtype(torch.float64)
device = torch.device('cpu')

def get_dataset(dataset_name, scale, batch_size):
    torch.manual_seed(0)
    np.random.seed(0)

    if dataset_name == "synthetic-classification-1000x1000":
        n = 1000
        d = 1000

        train_data = np.random.randn(n, d)
        train_data = (train_data + train_data.T) / 2
        w_star = np.random.randn(d)

        train_target = train_data @ w_star
        train_target[train_target < 0.0] = -1.0
        train_target[train_target > 0.0] = 1.0

        train_data = torch.Tensor(train_data)
        train_target = torch.Tensor(train_target)

        train_load = TensorDataset(train_data, train_target)
        train_dataloader = DataLoader(train_load, batch_size=batch_size, shuffle=False)

        r1 = -scale
        r2 = scale
        scaling_vec = (r1 - r2) * torch.rand(train_data.shape[1]) + r2
        scaling_vec = torch.pow(torch.e, scaling_vec)
        train_data_scaled = scaling_vec * train_data

        train_load_scaled = TensorDataset(train_data_scaled, train_target)
        train_dataloader_scaled = DataLoader(train_load_scaled, batch_size=batch_size, shuffle=False)

        train = train_data, train_target, train_dataloader
        train_scaled = train_data_scaled, train_target, train_dataloader_scaled
    else:
        datasets_path = os.getenv("LIBSVM_DIR")
        trainX, trainY = load_svmlight_file(f"{datasets_path}/{dataset_name}")

        train_data = torch.tensor(trainX.toarray())
        train_target = torch.tensor(trainY)

        train_load = TensorDataset(train_data, train_target)
        train_dataloader = DataLoader(train_load, batch_size=batch_size, shuffle=False)

        r1 = -scale
        r2 = scale
        scaling_vec = (r1 - r2) * torch.rand(train_data.shape[1]) + r2
        scaling_vec = torch.pow(torch.e, scaling_vec)
        train_data_scaled = scaling_vec * train_data

        train_load_scaled = TensorDataset(train_data_scaled, train_target)
        train_dataloader_scaled = DataLoader(train_load_scaled, batch_size=batch_size, shuffle=False)

        train = train_data, train_target, train_dataloader
        train_scaled = train_data_scaled, train_target, train_dataloader_scaled


    if scale == 0: 
        return train
    else:
        return train_scaled

opt_dict = {
    "adam": Adam,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
}

loss_dict = {
    "logreg": lf.logreg,
    "nllsq": lf.nllsq,
}

def main(dataset_name, scale, batch_size, epochs, loss, opt, n_trials, seed=0):

    optim = opt_dict.get(opt)
    loss_function = loss_dict.get(loss)
    data, target, dataloader = get_dataset(dataset_name, scale, batch_size)

    if loss_function in [lf.logreg]:
        target[target == target.unique()[0]] = torch.tensor(-1.0, dtype=torch.get_default_dtype())
        target[target == target.unique()[1]] = torch.tensor(1.0, dtype=torch.get_default_dtype())
        assert torch.equal(target.unique(), torch.tensor([-1.0, 1.0]))

    elif loss_function == lf.nllsq:
        target[target == target.unique()[0]] = 0.0
        target[target == target.unique()[1]] = 1.0
        assert torch.equal(target.unique(), torch.tensor([0.0, 1.0]))


    def objective(trial):
        torch.manual_seed(seed)
        w = torch.zeros(data.shape[1], device=device).requires_grad_() # parameters
        
        if scale == 0:
            lr = trial.suggest_float(name="lr", low=1e-4, high=5) # original
        else:
            lr = trial.suggest_float(name="lr", low=1e-7, high=1e-2) # scaled

        opt = optim([w], lr=lr)
        
        def compute_loss(w, data, target):
            loss = loss_function(w, data, target)
            loss.backward()
            return loss
        
        for epoch in range(epochs):
            for i, (batch_data, batch_target) in enumerate(dataloader):
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                opt.zero_grad()
                loss = compute_loss(w, batch_data, batch_target)
                opt.step()
        loss = loss_function(w, data.to(device), target.to(device))
        return loss

    study_name = f"{dataset_name}-{scale}-{epochs}-{batch_size}-{opt}-{loss}"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    print(f"Study: {study_name}")
    print(f"Best lr: {study.best_trial.params['lr']}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from datasets directory.")
    parser.add_argument("--scale", type=int, default=0, help="Scaling range. [-scale, scale].")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss", type=str, choices=["logreg", "nllsq", "mse"])
    parser.add_argument("--optimizer", type=str, choices=["adam", "adagrad", "adadelta"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()
    print(f"device: {device}")
    print(args)
    main(args.dataset, args.scale, args.batch_size, args.epochs, args.loss, args.optimizer, args.n_trials, args.seed)