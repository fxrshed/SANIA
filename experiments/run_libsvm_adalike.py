import os
from collections import defaultdict
import pickle
import argparse


import numpy as np

import utils
from loss_functions import LogisticRegressionLoss
from methods import *

import neptune

from dotenv import load_dotenv
load_dotenv()

def train_loop(dataset_name: str, dataset: list, scale: int, batch_size: int, n_epochs: int,
               optimizer: BaseOptimizer, lr: float, lmd: float = 0.0, seed: int = 0,
               neptune_mode: str = "async") -> dict: 
    
    np.random.seed(seed)
    
    train_data, train_target, test_data, test_target = dataset
    
    # parameters
    params = np.zeros(train_data.shape[1])
    optim = optimizer(params=params, lr=lr)

    # oracle 
    loss_function = LogisticRegressionLoss(lmd=lmd)
    
    # e.g. libsvm dataset has {0.0, 1.0} classes that cannot be used for LogisticRegressionLoss 
    # hence they will be remapped to {-1.0, 1.0}        
    if isinstance(loss_function, LogisticRegressionLoss):
        if not np.array_equal(np.unique(train_target), (-1.0, 1.0)):
            train_target = utils.map_classes_to(train_target, (-1.0, 1.0))
            test_target = utils.map_classes_to(test_target, (-1.0, 1.0))
    
    assert np.array_equal(np.unique(train_target), (-1.0, 1.0))
    
    # logging 
    history = defaultdict(list)

    indices = np.arange(train_data.shape[0])
    
    run = neptune.init_run(
        mode=neptune_mode,
        tags=["binary-classification", "linear"],
    )
    
    run["dataset"] = {
        "dataset_name": dataset_name,
        "train_batch_size": batch_size,
        "test_batch_size": test_data.shape[0], 
        "scale": scale,
    }
    run["n_epochs"] = n_epochs
    run["seed"] = seed
    run["optimizer/parameters"] = {
        "name": optimizer.__name__,
        **optim.defaults
    }
    run["loss_fn"] = {
        "name": loss_function.__class__.__name__,
        **loss_function.__dict__,
    }
    run["model"] = "linear"
    run["device"] = "cpu"
    
    for epoch in range(n_epochs):
        
        # Testing 
        test_loss, test_grad, test_acc = loss_function.func_grad_acc(params, test_data, test_target)
        test_g_norm = np.linalg.norm(test_grad)**2
        history["test/loss"].append(test_loss)
        history["test/acc"].append(test_acc)
        history["test/grad_norm"].append(test_g_norm)
        
        run["test/loss"].append(test_loss)
        run["test/acc"].append(test_acc)
        run["test/grad_norm"].append(test_g_norm)
        

        # Training 
        np.random.shuffle(indices)

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        train_epoch_grad_norm = 0.0
        
        for idx in range(train_data.shape[0]//batch_size):
            batch_indices = indices[idx*batch_size:(idx+1)*batch_size]
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices] 
        
            train_loss, train_grad, train_acc = loss_function.func_grad_acc(params, batch_data, batch_target)
            
            g_norm = np.linalg.norm(train_grad)**2
            
            optim.step(loss=train_loss, grad=train_grad)

            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            train_epoch_grad_norm += g_norm
            
            history["train/batch/loss"].append(train_loss)
            history["train/batch/acc"].append(train_acc)
            history["train/batch/grad_norm"].append(g_norm)
            
            
        train_epoch_loss = train_epoch_loss / (idx + 1)
        train_epoch_acc = train_epoch_acc / (idx + 1)
        train_epoch_grad_norm = train_epoch_grad_norm / (idx + 1)
        
        history["train/epoch/loss"].append(train_epoch_loss)
        history["train/epoch/acc"].append(train_epoch_acc)
        history["train/epoch/grad_norm"].append(train_epoch_grad_norm)
        
        run["train/loss"].append(train_epoch_loss)
        run["train/acc"].append(train_epoch_acc)
        run["train/grad_norm"].append(train_epoch_grad_norm)

        
    # Testing 
    test_loss, test_grad, test_acc = loss_function.func_grad_acc(params, test_data, test_target)
    test_g_norm = np.linalg.norm(test_grad)**2
    history["test/loss"].append(test_loss)
    history["test/acc"].append(test_acc)
    history["test/grad_norm"].append(test_g_norm)    
    history["params"].append(params)
    
    run["test/loss"].append(test_loss)
    run["test/acc"].append(test_acc)
    run["test/grad_norm"].append(test_g_norm)

    
    return history


optimizer_dict = {
    "Adam": Adam,
    "Adagrad": Adagrad,
}

def main(dataset_name: str, test_split: float, scale: int, 
         batch_size: int, n_epochs: int, optimizer: str, lr: float, lmd: float,
         neptune_mode:str, save: bool, seed: int):
    
    train_data, train_target, test_data, test_target = utils.get_libsvm(name=dataset_name, test_split=test_split)
    n_features = utils.datasets_params[dataset_name]["n_features"]

    dataset = train_data, train_target, test_data, test_target

    scaling_vec = None
    if scale > 0:
        scaling_vec = utils.generate_scaling_vec(size=n_features, scale=scale, seed=0)
        scaled_train_data = train_data.multiply(scaling_vec).tocsr()
        scaled_test_data = test_data.multiply(scaling_vec).tocsr()
        dataset = scaled_train_data, train_target, scaled_test_data, test_target
    
    if lr == -1.0:
        lrs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    else:
        lrs = [lr]
        
    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]
    
    for lr in lrs:
        for seed in seeds:
            hist = train_loop(dataset_name=dataset_name, 
                              dataset=dataset,
                              scale=scale,
                              batch_size=batch_size,
                              n_epochs=n_epochs,
                              optimizer=optimizer_dict[optimizer],
                              lr=lr, 
                              lmd=lmd,
                              seed=seed,
                              neptune_mode=neptune_mode)
            
            if save:
                utils.save_results(results=hist, model_name="linear", dataset_name=dataset_name,
                            scale=scale, batch_size=batch_size, n_epochs=n_epochs, 
                            optimizer=optimizer, lr=lr, seed=seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from LibSVM datasets directory.")
    parser.add_argument("--test_split", type=float, default=0.0, help="train-test split ratio.")
    parser.add_argument("--scale", type=int, default=0, help="Scaling range. [-scale, scale].")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--optimizer", type=str, choices=["Adam", "Adagrad"])
    parser.add_argument("--neptune_mode", type=str, default="async", choices=["async", "debug", "offline", "read-only", "sync"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--lmd", type=float, default=0.0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")
    
    args = parser.parse_args()
    print(args)
    
    main(dataset_name=args.dataset, test_split=args.test_split, scale=args.scale, 
        batch_size=args.batch_size, n_epochs=args.n_epochs, optimizer=args.optimizer, 
        lr=args.lr, lmd=args.lmd, neptune_mode=args.neptune_mode, save=args.save, seed=args.seed)