import os

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from sklearn.datasets import load_svmlight_file

from dotenv import load_dotenv
load_dotenv()


def get_MNIST(batch_size, percentage=1.0):
    
    datasets_path = os.getenv("DATASETS_DIR")

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
        ])

    trainset = torchvision.datasets.MNIST(root=datasets_path, train=True, download=True, transform=transform)
    random_indices = np.random.choice(trainset.data.shape[0], round(trainset.data.shape[0] * percentage), replace=False)
    
    trainset = torch.utils.data.Subset(trainset, random_indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.MNIST(root=datasets_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

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