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
# from torch_optimizer import Adahessian

from optimizer import PSPS2
import datasets
import loss_functions as lf
import models

from dotenv import load_dotenv
load_dotenv()

from utils import restricted_float

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizers_dict = {
    "adam": Adam,
    "adagrad": Adagrad,
    # "adahessian": Adahessian,
    "adadelta": Adadelta,
    "psps2": PSPS2,
}

models_dict = {
    "nn": models.NN,
    "lenet": models.LeNet5,
}


def main(model_name, optimizer_name, dataset_name, 
         batch_size, percentage, epochs, 
         seed=0, save=True, lr=1.0,
         precond_method=None, pcg_method=None,
         hutch_init_iters=1000):

    torch.random.manual_seed(seed)

    data, target, dataloader = datasets.get_MNIST(batch_size, percentage)
  
    loss_function = torch.nn.CrossEntropyLoss
    model = models_dict[model_name]

    if optimizer_name == "psps2":
        hist  = run_psps2(loss_function, data, target, dataloader, epochs, precond_method, 
                          pcg_method=pcg_method, hutch_init_iters=hutch_init_iters, scaling_vec=scaling_vec)
    elif optimizer_name == "sp2plus":
        hist = run_sp2plus(loss_function, data, target, dataloader, epochs)
    else:
        optimizer = optimizers_dict[optimizer_name]
        hist  = run_optimizer(optimizer, loss_function, data, target, dataloader, epochs, lr=lr)

    
    if save:
        save_results(hist, dataset_name, percentage, scale, batch_size, epochs, loss_function_name, 
                     optimizer_name, lr, precond_method, pcg_method, hutch_init_iters, seed)


def save_results(result, dataset_name, percentage, scale, batch_size, epochs, loss_function_name, optimizer_name, lr, 
                 precond_method, pcg_method, hutch_init_iters, seed):
    
    results_path = os.getenv("RESULTS_DIR")
    directory = f"{results_path}/{dataset_name}/percentage_{percentage}/scale_{scale}/bs_{batch_size}" \
    f"/epochs_{epochs}/{loss_function_name}/{optimizer_name}/lr_{lr}/precond_{precond_method}/pcg_method_{pcg_method}/hutch_init_iters_{hutch_init_iters}/seed_{seed}"

    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    torch.save([x[0] for x in result], f"{directory}/loss")
    torch.save([x[1] for x in result], f"{directory}/grad_norm_sq")
    torch.save([x[2] for x in result], f"{directory}/acc")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from datasets directory.")
    parser.add_argument("--percentage", type=restricted_float, default=1.0, help="What percentage of data to use. Range from (0.0, 1.0].")
    parser.add_argument("--scale", type=int, default=0, help="Scaling range. [-scale, scale].")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss", type=str, choices=["logreg", "nllsq", "mse"])
    parser.add_argument("--optimizer", type=str, choices=["psps2", "sp2plus", "adam", "adagrad", "adadelta"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--precond_method", type=str, choices=["none", "hutch", "adam", "adam_m", "adagrad", "adagrad_m", "pcg", "scaling_vec"], default="none")
    parser.add_argument("--pcg_method", type=str, choices=["none", "hutch", "adam", "adam_m", "adagrad", "adagrad_m"], default="none")
    parser.add_argument("--hutch_init_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    args = parser.parse_args()
    print(f"device: {device}")
    print(args)

    main(optimizer_name=args.optimizer, loss_function_name=args.loss, dataset_name=args.dataset, 
         batch_size=args.batch_size, percentage=args.percentage, scale=args.scale, 
         epochs=args.epochs, seed=args.seed, save=args.save, 
         lr=args.lr, precond_method=args.precond_method, pcg_method=args.pcg_method,
         hutch_init_iters=args.hutch_init_iters)