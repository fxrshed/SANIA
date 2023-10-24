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

from dotenv import load_dotenv
load_dotenv()

from utils import restricted_float

torch.set_default_dtype(torch.float64)

device = torch.device('cpu')





def run_sp2plus(loss_function, train_data, train_target, train_dataloader, epochs, lr=1.0):

    w = torch.zeros(train_data.shape[1], device=device).requires_grad_()
    # save loss and grad size to history
    hist = []

    eps = 1e-8
       
    for epoch in range(epochs):

        loss = loss_function(w, train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        grad_norm_sq = torch.linalg.norm(g) ** 2  
        acc = (np.sign(train_data @ w.detach().numpy()) == train_target).sum() / train_target.shape[0]

        # print(f"[{epoch}/{epochs}] | Loss: {loss.item()} | GradNorm^2: {grad_norm_sq.item()} | Accuracy: {acc}")
        hist.append([loss.item(), grad_norm_sq.item(), acc])
            

        for i, (batch_data, batch_target) in enumerate(train_dataloader):

            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            loss = loss_function(w, batch_data, batch_target)
            g, = torch.autograd.grad(loss, w, create_graph=True)
            f_grad = g.clone().detach()

            hgp = torch.autograd.grad(g, w, grad_outputs=g, retain_graph=True)[0]
            
            grad_norm_sq = torch.dot(f_grad, f_grad)
            polyak = loss / (grad_norm_sq + eps)
            v = f_grad - (hgp * polyak)
            v_norm_sq = torch.dot(v, v)
            step = (polyak * f_grad) + (0.5 * polyak**2 * (torch.dot(hgp, f_grad) / (v_norm_sq + eps )) * v) 

            with torch.no_grad():
                w.sub_(step, alpha=lr)

    return hist


def rademacher_old(weights):
    return torch.round(torch.rand_like(weights)) * 2 - 1

def diag_estimate_old(weights, grad, iters):
    Ds = []
    for j in range(iters):
        z = rademacher_old(weights)
        with torch.no_grad():
            hvp = torch.autograd.grad(grad, weights, grad_outputs=z, retain_graph=True)[0]
        Ds.append((hvp*z))

    return torch.mean(torch.stack(Ds), 0)

def run_psps2(loss_function, train_data, train_target, train_dataloader, epochs, precond_method, **kwargs):
    
    # parameters
    w = torch.zeros(train_data.shape[1], device=device).requires_grad_()

    # save loss and grad size to history
    hist = []
    

    loss = loss_function(w, train_data.to(device), train_target.to(device))
    g, = torch.autograd.grad(loss, w, create_graph=True)
    f_grad = g.clone().detach() 


    if precond_method == "none":
        D = torch.ones_like(w)
    elif precond_method == "hutch":
        alpha=0.1
        beta=0.999
        init_iters = kwargs["hutch_init_iters"]
        Dk = diag_estimate_old(w, g, init_iters)
    elif precond_method == "pcg":
        MAX_ITER = train_data.shape[1] * 2

    elif precond_method == "scaling_vec":
        scaling_vec = kwargs["scaling_vec"]
        D = (1 / scaling_vec)**2
    elif precond_method == "adam" or precond_method == "adam_m":
        D = torch.zeros_like(g)
        v = torch.zeros_like(g)
        step_t = torch.tensor(0.)
        betas = (0.9, 0.999)
    elif precond_method == "adagrad" or precond_method == "adagrad_m":
        D = torch.zeros_like(g)
        v = torch.zeros_like(g)

    pcg_method = kwargs.get("pcg_method")
    if pcg_method == "hutch":
        alpha=0.1
        beta=0.999
        init_iters = kwargs["hutch_init_iters"]
        Dk_pcg = diag_estimate_old(w, g, init_iters)
    elif pcg_method == "adam" or pcg_method == "adam_m":
        D_pcg = torch.zeros_like(g)
        v_pcg = torch.zeros_like(g)
        step_t_pcg = torch.tensor(0.)
        betas = (0.9, 0.999)
    elif pcg_method == "adagrad" or pcg_method == "adagrad_m":
        D_pcg = torch.zeros_like(g)
        v_pcg = torch.zeros_like(g)
    elif pcg_method == "none":
        D_pcg = torch.ones_like(g)


    for epoch in range(epochs):

        loss = loss_function(w, train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        grad_norm_sq = torch.linalg.norm(g) ** 2  
        acc = (np.sign(train_data @ w.detach().numpy()) == train_target).sum() / train_target.shape[0]

        # print(f"[{epoch}/{epochs}] | Loss: {loss.item()} | GradNorm^2: {grad_norm_sq.item()} | Accuracy: {acc}")
        hist.append([loss.item(), grad_norm_sq.item(), acc])
           
        for i, (batch_data, batch_target) in enumerate(train_dataloader):  

            loss = loss_function(w, batch_data, batch_target)
            g, = torch.autograd.grad(loss, w, create_graph=True)
            f_grad = g.detach().clone()

            if precond_method == "scaling_vec":
                s = D * f_grad

            elif precond_method == "adam" or precond_method == "adam_m":
                step_t += 1
                v = betas[1] * v + (1 - betas[1]) * g.square()
                v_hat = v / (1 - torch.pow(betas[1], step_t))

                if precond_method == "adam":
                    D = 1 / (torch.sqrt(v_hat) + 1e-6)
                elif precond_method == "adam_m":
                    D = 1 / (v_hat + 1e-6) 
                s = D * f_grad

            elif precond_method == "adagrad" or precond_method == "adagrad_m":
                v.add_(torch.square(g))
                if precond_method == "adagrad":
                    D = 1 / (torch.sqrt(v) + 1e-6)
                elif precond_method == "adagrad_m":
                    D = 1 / (v + 1e-6)
                s = D * f_grad

            elif precond_method == "none":
                s = D * f_grad

            elif precond_method == "hutch":
                vk = diag_estimate_old(w, g, 1)

                # Smoothing and Truncation 
                Dk = beta * Dk + (1 - beta) * vk
                Dk_hat = torch.abs(Dk)
                Dk_hat[Dk_hat < alpha] = alpha

                D = 1 / Dk_hat
                s = D * f_grad

            elif precond_method == "pcg":

                if pcg_method == "hutch":
                    vk_pcg = diag_estimate_old(w, g, 1)
                    # Smoothing and Truncation 
                    Dk_pcg = beta * Dk_pcg + (1 - beta) * vk_pcg
                    Dk_hat = torch.abs(Dk_pcg)
                    Dk_hat[Dk_hat < alpha] = alpha
                    D_pcg = 1 / Dk_hat

                elif pcg_method == "adam" or pcg_method == "adam_m":
                    step_t_pcg += 1
                    v_pcg = betas[1] * v_pcg + (1 - betas[1]) * f_grad.square()
                    v_hat = v_pcg / (1 - torch.pow(betas[1], step_t_pcg))
                    if pcg_method == "adam":
                        D_pcg = 1 / (torch.sqrt(v_hat) + 1e-6)
                    else:
                        D_pcg = 1 / (v_hat + 1e-6)

                elif pcg_method == "adagrad" or pcg_method == "adagrad_m":
                    v_pcg.add_(f_grad.square())
                    if pcg_method == "adagrad":
                        D_pcg = 1 / (torch.sqrt(v_pcg) + 1e-6)
                    else:
                        D_pcg = 1 / (v_pcg + 1e-6)

                hess_diag_inv = D_pcg.clone()
                # Preconditioned CG is here
                s = torch.zeros_like(w) # s = H_inv * grad
                r = f_grad.clone()
                z = hess_diag_inv * r
                p = z.detach().clone()

                for cg_step in range(MAX_ITER):
                    hvp = torch.autograd.grad(g, w, grad_outputs=p, retain_graph=True)[0]
                    alpha_k = torch.dot(r, z) / torch.dot(p, hvp)
                    s = s + alpha_k * p
                    r_prev = r.clone()
                    r = r - alpha_k * hvp
                    if torch.norm(r) < 1e-4:
                        break
                    
                    z_prev = z.clone()
                    z = hess_diag_inv * r
                    beta_k = torch.dot(r, z) / torch.dot(r_prev, z_prev)
                    p = z + beta_k * p    


            grad_norm_sq_scaled = torch.dot(f_grad, s)

            if 2 * loss <= grad_norm_sq_scaled:
                det = 1 - ( (2 * loss) / ( grad_norm_sq_scaled )) 
                step_size = 1 - torch.sqrt(det)
            else:
                # print(f"[{epoch}, {i}] No solution")
                step_size = 1.0

                
            with torch.no_grad():
                w.sub_(step_size * s)

    return hist




def run_optimizer(optimizer, loss_function, train_data, train_target, train_dataloader, epochs, **kwargs_optimizer):
    # parameters
    w = torch.zeros(train_data.shape[1], device=device).requires_grad_()
    opt = optimizer([w], **kwargs_optimizer)

    # logging 
    hist = []
    
    def compute_loss(w, data, target):
        loss = loss_function(w, data, target)
        loss.backward()
        return loss


    for epoch in range(epochs):

        loss = loss_function(w, train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        acc = (np.sign(train_data @ w.detach().numpy()) == train_target).sum() / train_target.shape[0]
        print(f"[{epoch}/{epochs}] | Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | Acc: {acc}")
        hist.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), acc])

        for i, (batch_data, batch_target) in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            opt.zero_grad()
            loss = compute_loss(w, batch_data, batch_target)
            opt.step()

    return hist



optimizers_dict = {
    "adam": Adam,
    "adagrad": Adagrad,
    # "adahessian": Adahessian,
    "adadelta": Adadelta,
    "psps2": PSPS2,
    "sgd": SGD,
    "rmsprop": RMSprop,
}

loss_functions_dict = {
    "logreg": lf.logreg,
    "nllsq": lf.nllsq,
}

def main(optimizer_name, loss_function_name, dataset_name, 
         batch_size, percentage, scale, 
         epochs, 
         seed=0, 
         save=True, 
         lr=1.0,
         precond_method=None,
         pcg_method=None,
         hutch_init_iters=1000):

    torch.random.manual_seed(seed)

    data, target, dataloader, scaling_vec = datasets.get_libsvm(dataset_name, batch_size, percentage, scale)
  
    loss_function = loss_functions_dict[loss_function_name]

    if loss_function == lf.logreg:
        target[target == target.unique()[0]] = torch.tensor(-1.0, dtype=torch.get_default_dtype())
        target[target == target.unique()[1]] = torch.tensor(1.0, dtype=torch.get_default_dtype())
        assert torch.equal(target.unique(), torch.tensor([-1.0, 1.0]))

    elif loss_function == lf.nllsq:
        target[target == target.unique()[0]] = 0.0
        target[target == target.unique()[1]] = 1.0
        assert torch.equal(target.unique(), torch.tensor([0.0, 1.0]))



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
    parser.add_argument("--optimizer", type=str, choices=["psps2", "sp2plus", "adam", "adagrad", "adadelta", "sgd"])
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