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
import Models
from Models import Conv2dBlock, CNNClassifier, RankConv2d, RankConv2dBlock
import utils
from utils import load_data, safe_mkdir, get_paths
import os

def run_exp_rank(DATA, DEVICE, kernels_size, ranks, seed = 42, ITERATIONS = 10000, BATCH_SIZE = 64, verbose = False, PRINT_EVERY = 100, OUT_CLASSES = 10):
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA
    parpath, folderpath = get_paths(kernels_size, ranks, seed)
    safe_mkdir(parpath)
    safe_mkdir(folderpath)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    layers = [RankConv2dBlock([3,32,32], [ranks,ranks], (kernels_size, kernels_size)), 
              RankConv2dBlock([32,64,64], [ranks,ranks], (kernels_size, kernels_size)), 
              RankConv2dBlock([64,128,128], [ranks,ranks], (kernels_size, kernels_size)), 
              nn.Flatten(), nn.Linear(2048,128), nn.Dropout(0.4), nn.ReLU(), nn.Linear(128,OUT_CLASSES)]
    MODEL = CNNClassifier(layers, 1e-3, nn.functional.cross_entropy).to(DEVICE)
    MODEL.train()

    print("DOING MODEL TRAINING, RANK: " + str(ranks) + ", KERNEL SIZE: " + str(kernels_size) + ", SEED: " + str(seed))
    print()
    
    losses = []
    train_start = time.time()

    for itr in range(ITERATIONS+1):
        idxs = np.random.randint(0,50000, size = (BATCH_SIZE,))

        X, Y = X_TRAIN[idxs], Y_TRAIN[idxs]
        
        loss = MODEL.backprop(X,Y)
        if itr%PRINT_EVERY == 0:
            if verbose:
                print(itr,loss)
        losses.append(loss)

    train_time = time.time() - train_start
    
    MODEL.eval()
    
    print("INTO EVAL MODE")

    #plt.clf()
    #plt.plot(np.arange(len(losses)), np.array(losses))
    #plt.title("Cross Entropy Loss over Training, KS " + str(kernels_size) + " RANK " + str(ranks) + " seed " + str(seed))
    #plt.savefig(folderpath + "/loss_graph.png")
    #plt.clf()
    
    print("PLOTTED")

    test_start = time.time()
    test_loss = MODEL.compute_loss(X_TEST, Y_TEST)
    test_acc = MODEL.compute_accuracy(X_TEST, Y_TEST)
    test_time = time.time() - test_start

    print("Testing Set Cross Entropy Loss:", test_loss)
    print("Testing Set Accuracy:", test_acc)
    
    train_loss, train_acc, t = 0, 0, 0
    for i in range(0,len(X_TRAIN) - BATCH_SIZE + 1,BATCH_SIZE):
        tl = MODEL.compute_loss(X_TRAIN[i:i+BATCH_SIZE], Y_TRAIN[i:i+BATCH_SIZE])
        ta = MODEL.compute_accuracy(X_TRAIN[i:i+BATCH_SIZE], Y_TRAIN[i:i+BATCH_SIZE])
        train_loss += tl
        train_acc += ta
        t += 1
    train_loss /= t
    train_acc /= t

    print("Training Set Cross Entropy Loss:", train_loss)
    print("Training Set Accuracy:", train_acc)

    print("TRAINING TIME:", train_time, "seconds")
    print("TEST TIME:",test_time, "seconds")
    
    np.savetxt(folderpath + "/losses.txt", np.array(losses))
    np.savetxt(folderpath + "/results_data.txt", np.array([test_loss, test_acc, test_time, train_loss, train_acc, train_time]))
    
    return losses, [test_loss, test_acc, test_time, train_loss, train_acc, train_time]

def run_full_exp_rank(DATA, DEVICE, kernels_size, ranks, num_seeds, ITERATIONS = 10000, BATCH_SIZE = 64, verbose = False, PRINT_EVERY = 100, OUT_CLASSES = 10):
    parpath = "Results/" + str(kernels_size) + "_" + str(ranks)
    full_losses, full_ex_data = [], []
    for seed in range(num_seeds):
        losses, ex_data = run_exp_rank(DATA, DEVICE, kernels_size, ranks, seed = seed, ITERATIONS = ITERATIONS, BATCH_SIZE = BATCH_SIZE, verbose = verbose, PRINT_EVERY = PRINT_EVERY, OUT_CLASSES = OUT_CLASSES)
        full_losses.append(losses)
        full_ex_data.append(ex_data)

    full_losses = np.array(full_losses)
    full_ex_data = np.array(full_ex_data)

    avg_losses = np.mean(full_losses, axis = 0)
    avg_ex_data = np.mean(full_ex_data, axis = 0)

    #plt.clf()
    #plt.plot(np.arange(len(avg_losses)), np.array(avg_losses))
    #plt.title("Average Cross Entropy Loss over Training, KS " + str(kernels_size) + " RANK " + str(ranks))
    #plt.savefig(parpath + "/avg_loss_graph.png")
    #plt.clf()

    print("Average Testing Set Cross Entropy Loss:", avg_ex_data[0])
    print("Average Testing Set Accuracy:", avg_ex_data[1])

    print("Average Training Set Cross Entropy Loss:", avg_ex_data[3])
    print("Average Training Set Accuracy:", avg_ex_data[4])

    print("Average TRAINING TIME:", avg_ex_data[5], "seconds")
    print("Average TEST TIME:", avg_ex_data[2], "seconds")

    np.savetxt(parpath + "/full_losses.txt", full_losses)
    np.savetxt(parpath + "/full_ex_data.txt", full_ex_data)

    return full_losses, full_ex_data
        
	