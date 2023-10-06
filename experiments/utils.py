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



