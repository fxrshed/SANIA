import os
import argparse
import re
import pickle
import datetime

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import numpy as np
import scipy

import svmlight_loader

from dotenv import load_dotenv
load_dotenv()

def save_results(results, model_name, dataset_name, scale, batch_size,
                 n_epochs, optimizer, lr, seed):

    results_path = os.getenv("RESULTS_DIR")

    directory = f"{results_path}/{model_name}/{dataset_name}/scale_{scale}/bs_{batch_size}" \
        f"/epochs_{n_epochs}/{optimizer}/lr_{lr}/seed_{seed}"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(f"{directory}/summary.p", "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {directory}")

def load_results(dataset_name: str, model_name: str, scale: int, batch_size: int, n_epochs: int, optimizer: str, lr: float, seed: int) -> dict:
    
    results_path = os.getenv("RESULTS_DIR")
    directory = f"{results_path}/{model_name}/{dataset_name}/scale_{scale}/bs_{batch_size}" \
        f"/epochs_{n_epochs}/{optimizer}/lr_{lr}/seed_{seed}"
    
    assert os.path.exists(directory), f"Results {directory} do not exist."

    # results = torch.load(f"{directory}/summary.p", map_location=torch.device('cpu'))
    
    with open(f"{directory}/summary.p", "rb") as f:
        results = pickle.load(f)

        
    return results 

def make_synthetic_binary_classification(
    n_samples: int, 
    n_features: int, 
    classes: tuple[int, int] = (-1, 1),
    symmetric: bool = False, 
    seed: int = 0):
    
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features)
    
    if symmetric:
        assert n_samples == n_features, f"`n_samples` must be equal to `n_features` to get symmetric matrix. " \
            f"Currently `n_samples={n_samples}` and `n_features={n_features}`."
        data = (data + data.T) / 2
    w_star = np.random.randn(n_features)

    target = data @ w_star
    target[target <= 0.0] = classes[0]
    target[target > 0.0] = classes[1]

    return data, target

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

def map_classes_to(target, new_classes):
    old_classes = np.unique(target)
    new_classes = np.sort(new_classes)
    
    if np.array_equal(old_classes, new_classes):
        return target
    
    assert np.unique(target).size == len(new_classes), \
        f"Old classes must match the number of new classes. " \
        f"Currently ({np.unique(target).size}) classes are being mapped to ({len(new_classes)}) new classes."

    mapping = {v: t for v, t in zip(old_classes, new_classes)}
    target = np.vectorize(mapping.get)(target)
    return target

def generate_scaling_vec(size: int = 0, scale: int = 1, seed: int = 0):
    
    np.random.seed(seed)
    
    r1 = -scale
    r2 = scale
    scaling_vec = (r1 - r2) * np.random.uniform(size=size) + r2
    scaling_vec = np.power(np.e, scaling_vec)

    return scaling_vec

datasets_path = os.getenv("LIBSVM_DIR")
datasets_params = {
    "webspam": {
        "train_path": f"{datasets_path}/webspam/webspam_train",
        "test_path": f"{datasets_path}/webspam/webspam_test",
        "n_features": 16_609_143, 
        },
    "news20.binary": {
        "train_path": f"{datasets_path}/news20/news20.binary",
        "n_features": 1_355_191, 
        },
    "a1a": {
        "train_path": f"{datasets_path}/a1a",
        "test_path": f"{datasets_path}/a1a.t",
        "n_features": 123,
    },
    "a8a": {
        "train_path": f"{datasets_path}/a8a",
        "test_path": f"{datasets_path}/a8a.t",
        "n_features": 123,
    },
    "a9a": {
        "train_path": f"{datasets_path}/a9a",
        "test_path": f"{datasets_path}/a9a.t",
        "n_features": 123,
    },
    "w1a": {
        "train_path": f"{datasets_path}/w1a",
        "test_path": f"{datasets_path}/w1a.t",
        "n_features": 300,
    },
    "w8a": {
        "train_path": f"{datasets_path}/w8a",
        "test_path": f"{datasets_path}/w8a.t",
        "n_features": 300,
    },
    "mushrooms": {
        "train_path": f"{datasets_path}/mushrooms",
        "n_features": 112,
    },
    "leu": {
        "train_path": f"{datasets_path}/leu",
        "test_path": f"{datasets_path}/leu.t",
        "n_features": 7129,
    },
    "real-sim": {
        "train_path": f"{datasets_path}/real-sim",
        "n_features": 20_958,
    },
    "rcv1.binary": {
        "train_path":  f"{datasets_path}/rcv1_train.binary",
        "test_path": f"{datasets_path}/rcv1_test.binary",
        "n_features": 47_236,
    },
    "colon-cancer": {
        "train_path": f"{datasets_path}/colon-cancer",
        "test_path": f"{datasets_path}/colon-cancer",
        "n_features": 2000,
    },
    "madelon": {
        "train_path": f"{datasets_path}/madelon",
        "test_path": f"{datasets_path}/madelon.t",
        "n_features": 500,
    },
    "gisette": {
        "train_path": f"{datasets_path}/gisette_scale",
        "test_path": f"{datasets_path}/gisette_scale.t",
        "n_features": 5000,
    },
    "duke": {
        "train_path": f"{datasets_path}/duke",
        "test_path": f"{datasets_path}/duke.tr",
        "n_features": 7129,
    },
    "diabetes_scale": {
        "train_path": f"{datasets_path}/diabetes_scale",
        "test_path": f"{datasets_path}/diabetes_scale",
        "n_features": 8,
    },
    "covtype.binary": {
        "train_path": f"{datasets_path}/covtype.libsvm.binary",
        "n_features": 54,
    },
    "covtype.binary.scale": {
        "train_path": f"{datasets_path}/covtype.libsvm.binary.scale",
        "n_features": 54,
    },
    "australian_scale": {
        "train_path": f"{datasets_path}/australian_scale",
        "test_path": f"{datasets_path}/australian_scale",
        "n_features": 14,
    },
    "breast-cancer_scale": {
        "train_path": f"{datasets_path}/breast-cancer_scale",
        "test_path": f"{datasets_path}/breast-cancer_scale",
        "n_features": 10,
    },
    "sonar_scale": {
        "train_path": f"{datasets_path}/sonar_scale",
        "test_path": f"{datasets_path}/sonar_scale",
        "n_features": 60,
    }
}


def get_libsvm(name: str, test_split: float = 0.0, seed: int = 0):
    
    test_data = None 
    test_target = None

    train_path = datasets_params[name].get("train_path")
    test_path = datasets_params[name].get("test_path")
    n_features = datasets_params[name]["n_features"]

    train_data, train_target = svmlight_loader.load_svmlight_file(train_path, n_features=n_features)
    
    if test_path is not None:
        test_data, test_target = svmlight_loader.load_svmlight_file(test_path, n_features=n_features)
    elif test_split > 0.0:
        print(f"Test data for `{name}` is not found. Splitting train into {(1 - test_split) * 100}% train and {test_split * 100}% test.")
        train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=test_split, random_state=seed)
        
    return train_data, train_target, test_data, test_target

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.01 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.01, 1.0]"%(x,))
    return x


def map_classes_to(target, new_classes):
    old_classes = np.unique(target)
    new_classes = np.sort(new_classes)
    
    if np.array_equal(old_classes, new_classes):
        return target
    
    assert np.unique(target).size == len(new_classes), \
        f"Old classes must match the number of new classes. " \
        f"Currently ({np.unique(target).size}) classes are being mapped to ({len(new_classes)}) new classes."

    mapping = {v: t for v, t in zip(old_classes, new_classes)}
    target = np.vectorize(mapping.get)(target)
    return target


# def F_and_grad_res(X, y, w):
#     r = np.exp( -X@w * y )
#     ry = -r/(1+r) * y
#     loss =  np.mean(np.log(1 + r ))
#     return loss, ry

# @numba.njit
# def update_params_sparse(params, grad_res, data, n_samples, indices, indptr, lr):
#     for i, gr in enumerate(grad_res):
#         row_start = indptr[i]
#         row_end = indptr[i + 1]
#         row_data = data[row_start:row_end]
#         grad = (row_data * gr) / n_samples
#         params[indices[row_start:row_end]] -= lr * grad

# np.random.seed(0)

# w = np.zeros(X.shape[1])
# lr = 0.5

# indices = list(range(X.shape[0]))
# batch_size = 512
# epochs = 200
# hist = []
# start = time.perf_counter()
# for e in range(epochs):
#     loss = lgstc(X, y, w)
#     # print(f"Epoch: {e}/{epochs} | Loss: {loss}")
#     hist.append(loss)
#     np.random.shuffle(indices)
#     for idx in range(X.shape[0] // batch_size):
#         batch_indices = indices[idx*batch_size:(idx+1)*batch_size]
#         batch_data = X[batch_indices]
#         batch_target = y[batch_indices] 
        
#         loss, grad_res = F_and_grad_res(batch_data, batch_target, w)
#         update_params_sparse(w, grad_res, batch_data.data, batch_data.shape[0], batch_data.indices, batch_data.indptr, lr)
    
# end = time.perf_counter()
# print(f"Time elapsed: {end - start}")


