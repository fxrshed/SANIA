import os
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from sklearn.datasets import load_svmlight_file

import experiments.loss_functions as lf

from dotenv import load_dotenv
load_dotenv()

from torch.optim import SGD, Adam, Adagrad, Adadelta, RMSprop


libsvm_namespace = ['mushrooms', 'colon-cancer', 'covtype.libsvm.binary', 'covtype.libsvm.binary.scale']

def save_results(results, model_name, dataset_name, scale, batch_size,
                 epochs, optimizer, lr, seed):

    results_path = os.getenv("RESULTS_DIR")
    directory = f"{results_path}/DNN/{dataset_name}/{model_name}/scale_{scale}/bs_{batch_size}" \
    f"/epochs_{epochs}/{optimizer}/lr_{lr}/seed_{seed}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    assert "train_hist" in results, "train_hist is empty."
    assert "test_hist" in results, "test_hist is empty."
    assert "model_state_dict" in results, "model_state_dict is empty."

    torch.save(results, f"{directory}/summary.pth")
    print(f"Results saved to {directory}")

def load_results(dataset_name: str, model_name: str, scale: int, batch_size: int, epochs: int, optimizer: str, lr: float, seed: int) -> dict:
    
    results_path = os.getenv("RESULTS_DIR")
    if model_name == "xxx":
        directory = f"{results_path}/DNN/{dataset_name}/scale_{scale}/bs_{batch_size}" \
            f"/epochs_{epochs}/{optimizer}/lr_{lr}/seed_{seed}"
    else:
        directory = f"{results_path}/DNN/{dataset_name}/{model_name}/scale_{scale}/bs_{batch_size}" \
            f"/epochs_{epochs}/{optimizer}/lr_{lr}/seed_{seed}"
    
    assert os.path.exists(directory), f"Results f{directory} do not exist."
    
    
    results = torch.load(f"{directory}/summary.pth", map_location=torch.device('cpu'))
    return results 










# def get_dataset(name, batch_size, percentage=1.0, scale=None):

#     datasets_path = os.getenv("DATASETS_DIR")
#     libsvm_path = os.getenv("LIBSVM_DIR")

#     if name == "MNIST":
#         assert scale == None, "Scaling not applicable."
#         train_dataset = torchvision.datasets.MNIST(root='./datasets', 
#                                                 train=True, 
#                                                 transform=transforms.ToTensor(),  
#                                                 download=True)
                                                
#         test_dataset = torchvision.datasets.MNIST(root='./datasets', 
#                                                 train=False, 
#                                                 transform=transforms.ToTensor()) 

#         # Data loader
#         train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                                 batch_size=batch_size, 
#                                                 shuffle=True)
#         test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                                 batch_size=batch_size, 
#                                                 shuffle=False) 
#         return train_loader, test_loader

#     else:

#         trainX, trainY = load_svmlight_file(f"{libsvm_path}/{name}")
#         sample = np.random.choice(trainX.shape[0], round(trainX.shape[0] * percentage), replace=False)
        
#         assert sample.shape == np.unique(sample).shape
        
#         trainX = trainX[sample]
#         trainY = trainY[sample]

#         train_data = torch.tensor(trainX.toarray(), dtype=torch.float)
#         train_target = torch.tensor(trainY, dtype=torch.float)

#     if scale != None:
#         r1 = -scale
#         r2 = scale
#         scaling_vec = (r1 - r2) * torch.rand(train_data.shape[1]) + r2
#         scaling_vec = torch.pow(torch.e, scaling_vec)
#         train_data = scaling_vec * train_data

#     return train_data, train_target


losses_dict = {
    "logreg": lf.logreg, 
    "nllsq": lf.nllsq
}

optimizers_dict = {
    "sgd": SGD,
    "adam": Adam,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
    "rmsprop": RMSprop,
}


def get_dataset(dataset_name, percentage, scale):

    datasets_path = os.getenv("LIBSVM_DIR")
    trainX, trainY = load_svmlight_file(f"{datasets_path}/{dataset_name}")
    sample = np.random.choice(trainX.shape[0], round(trainX.shape[0] * percentage), replace=False)

    assert sample.shape == np.unique(sample).shape

    trainX = trainX[sample]
    trainY = trainY[sample]

    train_data = torch.tensor(trainX.toarray(), dtype=torch.float)
    train_target = torch.tensor(trainY, dtype=torch.float)

    r1 = -scale
    r2 = scale
    scaling_vec = (r1 - r2) * torch.rand(train_data.shape[1]) + r2
    scaling_vec = torch.pow(torch.e, scaling_vec)
    train_data_scaled = scaling_vec * train_data

    return train_data_scaled, train_target, scaling_vec

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.01 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.01, 1.0]"%(x,))
    return x



