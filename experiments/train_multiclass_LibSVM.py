import os
from collections import defaultdict
import pickle
import argparse


import numpy as np
import scipy

import utils
from loss_functions import CrossEntropyLoss
from methods import *

import neptune

from dotenv import load_dotenv
load_dotenv()

def one_hot_encode(y, n_classes):
    one_hot_matrix = np.zeros((y.shape[0], n_classes))
    one_hot_matrix[np.arange(y.shape[0]), y] = 1
    return one_hot_matrix

def predict(data, params):
    z = data.dot(params)
    y_pred = scipy.special.softmax(z, axis=1)
    return np.argmax(y_pred, axis=1)

def train_loop(dataset_name: str, scale: float, dataset: list, batch_size: int, n_epochs: int,
               optimizer_name: str, lr: float, eps: float, lmd: float, neptune_mode: str, seed: int) -> dict: 
    
    np.random.seed(seed)
    
    train_data, train_target, test_data, test_target = dataset
    
    # parameters
    n_features = train_data.shape[1]
    n_classes = len(np.unique(train_target))

    params = np.zeros((n_features, n_classes))
    
    if optimizer_name in ["Adam", "Adagrad"]:
        optimizer = optimizer_dict[optimizer_name](params=params, lr=lr)
    else:
        optimizer = optimizer_dict[optimizer_name](params=params, lr=lr, eps=eps)

    # oracle 
    loss_function = CrossEntropyLoss()

    # logging 
    history = defaultdict(list)

    indices = np.arange(train_data.shape[0])
    
    run = neptune.init_run(
        mode=neptune_mode,
        tags=["multiclass-classification", "linear"],
    )
    
    run["dataset"] = {
        "dataset_name": dataset_name,
        "train_batch_size": batch_size,
        "test_batch_size": test_data.shape[0], 
        "scale": scale,
    }
    run["n_epochs"] = n_epochs
    run["seed"] = seed
    
    run["loss_fn"] = {
        "name": loss_function.__class__.__name__,
        **loss_function.__dict__,
    }
    run["model"] = "linear"
    run["device"] = "cpu"
    
    run["optimizer/parameters/name"] = optimizer_name
    run["optimizer/parameters"] = neptune.utils.stringify_unsupported(optimizer.defaults)
   
    for epoch in range(n_epochs):
        
        true = test_target.astype(int) - 1
        pred = predict(test_data, params)
        
        y_true = one_hot_encode(true, n_classes)
        y_pred = one_hot_encode(pred, n_classes)
        loss = loss_function.func(y_true, y_pred)
        accuracy = (true == pred).mean()

        history["test/loss"].append(loss)
        history["test/acc"].append(accuracy)
        
        run["test/loss"].append(loss)
        run["test/acc"].append(accuracy)
        
        np.random.shuffle(indices)
        
        for idx in range(train_data.shape[0]//batch_size):
            batch_indices = indices[idx*batch_size:(idx+1)*batch_size]
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices] 
        
            y_true = one_hot_encode(batch_target.astype(int) - 1, n_classes)
            y_pred = one_hot_encode(predict(batch_data, params), n_classes)
            
            train_loss = loss_function.func(y_true, y_pred)
            train_grad = loss_function.grad(batch_data, y_true, y_pred)
            
            optimizer.step(loss=train_loss, grad=train_grad)
            
    print(f"After {n_epochs} epochs: Loss: {loss:.4f} | Acc: {accuracy:.2f}")

    return history


optimizer_dict = {
    "SANIA_AdamSQR": SANIA_AdamSQR,
    "SANIA_AdagradSQR": SANIA_AdagradSQR,
    "Adam": Adam,
    "Adagrad": Adagrad,
}

def main(seed: int, dataset_name: str, test_split: float, scale: int, 
         batch_size: int, n_epochs: int, optimizer_name: str, lr: float, eps: float, lmd: float, save: bool,
         neptune_mode: str):
    
    train_data, train_target, test_data, test_target = utils.get_libsvm(name=dataset_name, test_split=test_split)
    n_features = utils.datasets_params[dataset_name]["n_features"]
    
    if test_split == 0.0:
        test_data, test_target = train_data, train_target

    dataset = train_data, train_target, test_data, test_target

    scaling_vec = None
    if scale > 0:
        scaling_vec = utils.generate_scaling_vec(size=n_features, scale=scale, seed=0)
        scaled_train_data = train_data.multiply(scaling_vec).tocsr()
        scaled_test_data = test_data.multiply(scaling_vec).tocsr()
        dataset = scaled_train_data, train_target, scaled_test_data, test_target
    
    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]
    
    for seed in seeds:
        hist = train_loop(dataset_name=dataset_name,
                        scale=scale,
                        dataset=dataset,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        optimizer_name=optimizer_name, 
                        lr=lr,
                        eps=eps,
                        lmd=lmd,
                        neptune_mode=neptune_mode,
                        seed=seed)

        if save:
            utils.save_results(results=hist, model_name="linear", dataset_name=dataset_name,
                        scale=scale, batch_size=batch_size, n_epochs=n_epochs, 
                        optimizer=optimizer_name, lr=lr, seed=seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from LibSVM datasets directory.")
    parser.add_argument("--test_split", type=float, default=0.0, help="train-test split ratio.")
    parser.add_argument("--scale", type=int, default=0, help="Scaling range. [-scale, scale].")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--optimizer", type=str, choices=["Adam", "Adagrad", "SANIA_AdamSQR", "SANIA_AdagradSQR"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--lmd", type=float, default=0.0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")
    parser.add_argument("--neptune_mode", type=str, default="async", choices=["async", "debug", "offline", "read-only", "sync"])

    args = parser.parse_args()
    print(args)
    
    main(seed=args.seed, 
        dataset_name=args.dataset, 
        test_split=args.test_split,
        scale=args.scale, 
        batch_size=args.batch_size, 
        n_epochs=args.n_epochs, 
        optimizer_name=args.optimizer, 
        lr=args.lr, 
        eps=args.eps, 
        lmd=args.lmd, 
        save=args.save, 
        neptune_mode=args.neptune_mode)