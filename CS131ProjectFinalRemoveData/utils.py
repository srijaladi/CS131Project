import torch
import torchvision
import cifar10
import torchvision.datasets as D
import numpy as np
import torch.nn as nn
import matplotlib
from matplotlib import pyplot as plt
import time
import os

def load_data(DEVICE):
    cifar_train = D.CIFAR10(root='./data', train=True, download=True, transform=None)
    cifar_test = D.CIFAR10(root='./data', train=False, download=True, transform=None)
    print(cifar_train.data.shape)
    print(cifar_test.data.shape)

    Y_TRAIN = torch.from_numpy(np.array(cifar_train.targets)).to(DEVICE).long()
    Y_TEST = torch.from_numpy(np.array(cifar_test.targets)).to(DEVICE).long()
    X_TRAIN = torch.from_numpy(np.array(cifar_train.data)).to(DEVICE).float()
    X_TEST = torch.from_numpy(np.array(cifar_test.data)).to(DEVICE).float()

    X_TRAIN = torch.swapaxes(torch.swapaxes(X_TRAIN, 2, 3), 1, 2)/255
    X_TEST = torch.swapaxes(torch.swapaxes(X_TEST, 2, 3), 1, 2)/255
    
    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_paths(kernels_size, ranks, seed):
    parpath = "Results/" + str(kernels_size) + "_" + str(ranks)
    folderpath = parpath + "/" + str(seed)
    return parpath, folderpath