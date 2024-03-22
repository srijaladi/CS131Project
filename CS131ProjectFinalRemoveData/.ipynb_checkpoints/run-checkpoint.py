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
import Experiments
from Experiments import run_exp_rank, run_full_exp_rank
import utils
from utils import load_data, safe_mkdir, get_paths
import os
import sys
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ks",
    type=int,
    required = True,
    default=3,
    help="Kernels sizes to use during convolutions",
)

parser.add_argument(
    "--rank",
    type=int,
    required = True,
    default=1,
    help="Rank to project kernels/weights onto during training",
)

parser.add_argument(
    "--num_seeds",
    type=int,
    required = True,
    default=100,
    help="Number of seeds to test on and aggregate",
)

parser.add_argument(
    "--iters",
    type=int,
    required = False,
    default=10000,
    help="Number of iterations to train for",
)

parser.add_argument(
    "--batch_size",
    type=int,
    required = False,
    default=64,
    help="Batch size to train with during each iteration",
)

parser.add_argument(
    "--verbose",
    type=bool,
    required = False,
    default=False,
    help="Whether or not to print model loss during each print_every during training",
)

parser.add_argument(
    "--print_every",
    type=int,
    required = False,
    default=100,
    help="How often to print the model's training loss as a checkpoint",
)

parser.add_argument(
    "--out_classes",
    type=int,
    required = False,
    default=10,
    help="Total number of possible output classes to classify from",
)

args = parser.parse_args()

if __name__ == "__main__":
    print("BEGINNING EXPERIMENTS")
    out = run_full_exp_rank(load_data(DEVICE), DEVICE, args.ks, args.rank, num_seeds = args.num_seeds, ITERATIONS = args.iters, BATCH_SIZE = args.batch_size, verbose = args.verbose, PRINT_EVERY = args.print_every, OUT_CLASSES = args.out_classes)