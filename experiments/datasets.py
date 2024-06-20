import os
import re

import torch
import torchvision
from torchvision.transforms import v2

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

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
    
    transforms = v2.Compose([
        v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.1307,), (0.3081,),
        ),
    ])

    train_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms)


    test_data = torchvision.datasets.MNIST(TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms)

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader



def get_CIFAR10(train_batch_size: int, test_batch_size: int, seed: int = 0) -> tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(32, 32), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261),
        ),
    ])

    train_data = torchvision.datasets.CIFAR10(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms
        )

    test_data = torchvision.datasets.CIFAR10(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms
        )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_CIFAR100(train_batch_size: int, test_batch_size: int, seed: int = 0) -> tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(32, 32), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261),
        ),
    ])

    train_data = torchvision.datasets.CIFAR100(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms
        )

    test_data = torchvision.datasets.CIFAR100(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms
        )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

    
    
def get_FashionMNIST(train_batch_size: int, test_batch_size: int, seed: int = 0) -> Tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(28, 28), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize(
            (0.5,), (0.5,),
        ),
    ])

    train_data = torchvision.datasets.FashionMNIST(
        TORCHVISION_DATASETS_DIR, train=True, download=True, transform=transforms
        )

    test_data = torchvision.datasets.FashionMNIST(
        TORCHVISION_DATASETS_DIR, train=False, download=True, transform=transforms
        )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader