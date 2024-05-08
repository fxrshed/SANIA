import os

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from sklearn.datasets import load_svmlight_file

from dotenv import load_dotenv
load_dotenv()

from typing import Tuple

TORCHVISION_DATASETS_DIR = os.getenv("TORCHVISION_DATASETS_DIR")

# def get_MNIST(batch_size, percentage=1.0):
    
#     datasets_path = os.getenv("DATASETS_DIR")

#     transform = transforms.Compose([
#         transforms.Resize((32,32)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = (0.1307,), std = (0.3081,))
#         ])

#     trainset = torchvision.datasets.MNIST(root=datasets_path, train=True, download=True, transform=transform)
#     random_indices = np.random.choice(trainset.data.shape[0], round(trainset.data.shape[0] * percentage), replace=False)
    
#     trainset = torch.utils.data.Subset(trainset, random_indices)
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
#     testset = torchvision.datasets.MNIST(root=datasets_path, train=False, download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

#     return train_loader, test_loader


class CustomScaleTransform(object):
    
    def __init__(self, scaling_matrix):
        
        self.scaling_matrix = scaling_matrix
        
    def __call__(self, img):
        
        img_squeezed = img.squeeze()
        transformed_img_tensor = self.scaling_matrix * img_squeezed

        return transformed_img_tensor.unsqueeze(dim=0)
    
    
def get_MNIST(train_batch_size: int, test_batch_size: int, scale: int = 0, seed: int = 0) -> Tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    
    n_features = (28, 28)
    uniform_matrix = (-scale - scale) * torch.rand(n_features) + scale
    scaling_matrix = torch.exp(uniform_matrix)

    train_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,)),
                                CustomScaleTransform(scaling_matrix=scaling_matrix),
                                ]))


    test_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,)),
                                CustomScaleTransform(scaling_matrix=scaling_matrix),
                                ]))

    # train_length = 6000
    # test_length = 1000
    # subsample_train_indices = torch.randperm(len(train_data))[:train_length]
    # subsample_test_indices = torch.randperm(len(test_data))[:test_length]

    # train_loader = DataLoader(train_data, batch_size=batch_size_train, sampler=SubsetRandomSampler(subsample_train_indices))
    # test_loader = DataLoader(test_data, batch_size=batch_size_test, sampler=SubsetRandomSampler(subsample_test_indices))

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader



def get_CIFAR10(train_batch_size: int, test_batch_size: int, seed: int = 0) -> Tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)

    train_data = torchvision.datasets.CIFAR10(TORCHVISION_DATASETS_DIR, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                ]))


    test_data = torchvision.datasets.CIFAR10(TORCHVISION_DATASETS_DIR, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                ]))

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
    
    
def get_FashionMNIST(train_batch_size: int, test_batch_size: int, seed: int = 0) -> Tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)

    train_data = torchvision.datasets.FashionMNIST(TORCHVISION_DATASETS_DIR, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.5,), (0.5,)),
                                ]))


    test_data = torchvision.datasets.FashionMNIST(TORCHVISION_DATASETS_DIR, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.5,), (0.5,)),
                                ]))

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader




def get_libsvm(name, batch_size, percentage, scale=0, shuffle=True):

    datasets_path = os.getenv("LIBSVM_DIR")
    trainX, trainY = load_svmlight_file(f"{datasets_path}/{name}")
    
    sample = np.random.choice(trainX.shape[0], round(trainX.shape[0] * percentage), replace=False)

    assert sample.shape == np.unique(sample).shape

    trainX = trainX[sample]
    trainY = trainY[sample]

    train_data = torch.tensor(trainX.toarray())
    train_target = torch.tensor(trainY)

    r1 = -scale
    r2 = scale
    scaling_vec = (r1 - r2) * torch.rand(train_data.shape[1]) + r2
    scaling_vec = torch.pow(torch.e, scaling_vec)
    train_data = scaling_vec * train_data
    train_load = TensorDataset(train_data, train_target)
    train_dataloader = DataLoader(train_load, batch_size=batch_size, shuffle=shuffle)

    return train_data, train_target, train_dataloader, scaling_vec