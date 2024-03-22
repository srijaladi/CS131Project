import torch
import torchvision
import cifar10
import torchvision.datasets as D
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import matplotlib
from matplotlib import pyplot as plt
import time
import utils
from utils import load_data, safe_mkdir, get_paths
import os

class Conv2dBlock(nn.Module):
    def __init__(self, channels, kernel_size = (3,3), padding = 'same', pool_size = (2,2), pool_stride = (2,2)):
        super().__init__()
        assert(len(channels) >= 2)
        self.layers = []
        for i in range(1,len(channels)):
            prev_c, curr_c = channels[i-1], channels[i]
            self.layers.append(nn.Conv2d(in_channels = prev_c, out_channels = curr_c, kernel_size = kernel_size, padding = padding))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(pool_size, pool_stride))
        self.model = nn.Sequential(*self.layers)
    def forward(self,x):
        return self.model(x)

class CNNClassifier(nn.Module):
    def __init__(self, layers, learning_rate, loss_func):
        super().__init__()
        self.layers = layers
        self.model = nn.Sequential(*layers)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.loss_func = loss_func
    def forward(self,x):
        if len(x.size()) == 3:
            return self.model(x.unsqueeze(0)).squeeze(0)
        return self.model(x)
    def compute_loss(self,x,y):
        out = self.forward(x)
        loss = self.loss_func(out,y)
        return loss.detach().item()
    def backprop(self,x,y):
        self.optim.zero_grad()
        out = self.forward(x)
        loss = self.loss_func(out,y)
        loss.backward()
        self.optim.step()
        return loss.detach().item()
    def predict(self,x):
        out = self.forward(x)
        z = torch.argmax(out, dim = 1).squeeze()
        return z
    def compute_accuracy(self,x,y):
        z = self.predict(x)
        assert(y.size() == z.size())
        return (torch.sum(y == z)/len(y)).detach().item()
    
class RankConv2d(nn.Module):
    def __init__(self, rank, in_channels, out_channels, kernel_size = (3,3), padding = 'same'):
        super().__init__()
        assert(rank <= kernel_size[0] and rank <= kernel_size[1])
        self.in_c = in_channels
        self.out_c = out_channels
        self.rank = rank
        self.padding = padding
        self.last_training = True
        self.L = None
        self.R = None
        
        assert(len(kernel_size) == 2 and kernel_size[0] == kernel_size[1])
        self.full_rank = bool(rank == kernel_size[0])
        
        W0 = torch.empty(self.out_c, self.in_c, kernel_size[0], kernel_size[1])
        nn.init.xavier_uniform_(W0)
        U, S, V = torch.linalg.svd(W0)
        
        self.W, self.left_W, self.right_W = None, None, None
        if self.full_rank:
            self.W = Parameter(W0)
            self.W.requires_grad = True
        else:
            self.left_W = Parameter(torch.sqrt(S[:,:,:rank]).unsqueeze(2) * U[:,:,:,:rank])
            self.right_W = Parameter(torch.sqrt(S[:,:,:rank]).unsqueeze(2) * V[:,:,:,:rank])
            self.left_W.requires_grad = True
            self.right_W.requires_grad = True
                
        self.bias = Parameter(torch.empty(self.out_c,))
        nn.init.zeros_(self.bias)
        self.bias.requires_grad = True
        
    def create_toeplitz(self, V, D):
        assert len(V.size() == 2
        assert D >= V.size(1)
        res = torch.zeros((V.size(0), D - V.size(1) + 1, D))
        for i in range(D):
            res[:,i,i:i+V.size(1)] = V
        return res
        
    def compute_toeplitz_prod(self, M, D):
        assert len(M.size() == 3)
        res = self.create_toeplitz(M.select(-1,0), D)
        for i in range(1,M.size(-1)):
            T = self.create_toeplitz(M.select(-1, i), D)
            res = res @ T
        return res
        
    def compute_LR(self, M_L, M_R, D):
        L, R = self.compute_toeplitz_prod(M_L, D), self.compute_toeplitz_prod(M_R, D)
        return L, R

    def custom_conv2d(self, L, R, X, B):
        if self.padding == "same":
            Xp = torch.nn.functional.pad(X, (self.rank//2, self.rank//2, self.rank//2, self.rank//2), "constant", 0)
        else:
            Xp = X
        res = (L @ Xp) @ R + B
        return res
        
    def forward(self, x):
        if self.training and not(self.last_training):
            self.L, self.R = self.compute_LR(self.left_W, self.right_W, 2 * (x.size(-1)//2) + x.size(-1))
            
        if self.full_rank:
            return torch.nn.functional.conv2d(x, W, bias = self.bias, padding = self.padding)
        else:
            if not(self.training):
                L, R = self.compute_LR(self.left_W, self.right_W, x.size(-1) + (self.left_W.size(-1)//2) * 2)
            else:
                L, R = self.L, self.R
            return self.custom_conv2d(L, R, x, self.bias, 'same')
    
class RankConv2dBlock(nn.Module):
    def __init__(self, channels, ranks, kernel_size = (3,3), padding = 'same', pool_size = (2,2), pool_stride = (2,2)):
        super().__init__()
        assert(len(channels) >= 2 and len(ranks) + 1 == len(channels))
        self.layers = []
        for i in range(1,len(channels)):
            prev_c, curr_c = channels[i-1], channels[i]
            self.layers.append(RankConv2d(ranks[i-1], in_channels = prev_c, out_channels = curr_c, kernel_size = kernel_size, padding = padding))
            self.layers.append(nn.BatchNorm2d(curr_c))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(pool_size, pool_stride))
        self.model = nn.Sequential(*self.layers)
    def forward(self,x):
        return self.model(x)