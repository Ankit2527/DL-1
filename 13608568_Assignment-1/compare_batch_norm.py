################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import json

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [20,10]

# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    # TODO: Run all hyperparameter configurations as requested
    
    hidden_dims_list = [[128], [256, 128], [512, 256, 128]]
    epochs = 20
    lr = 0.1
    batch_size = 128
    seed = 42
    data_dir = './data'
    path = './'
    batch_norm_use = [True, False]
    d1 = {}
    if not(os.path.exists(path)):
        os.makedirs(path)
    
    
    for dims in hidden_dims_list:
        for use in batch_norm_use:
            d2 = {}
            d3 = str(use)
    
            print('\n')
            print('Model with hidden layers', dims)
            best_model, train_loss, valid_loss, train_accuracy, validation_accuracy = train_mlp_pytorch.train(hidden_dims = dims,lr = lr, batch_size = batch_size, 
                                                                            epochs = epochs, seed = seed, data_dir = data_dir, 
                                                                            use_batch_norm = use, save_graph = False)
            d2['train_loss' + d3] = train_loss
            d2['valid_loss' + d3] = valid_loss
            d2['train_accuracy' + d3] = train_accuracy
            d2['validation_accuracy' + d3] = validation_accuracy
            d1[str(dims) + d3] = d2
        
        
    results_filename = "Pytorch_MLP_Batch_Norm_result_file.json"
    results_filename = os.path.join(path, results_filename)
    with open(results_filename, "w") as f:
        json.dump(d1, f, indent = 4, sort_keys = True)
    print("Pytorch Batch Norm results saved")
    
    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    
    with open('Pytorch_MLP_Batch_Norm_result_file.json') as json_file:
        d1 = json.load(json_file)
        
    hidden_dims_list = [[128], [256, 128], [512, 256, 128]]
    batch_norm_use = [True, False]
    
    for use in batch_norm_use:
        path = './'
        d3 = str(use)
        
        plt.title('PyTorch_MLP_Loss_with_Batch_Norm = '+ d3)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
    
    
        for dims in hidden_dims_list:
            plt.plot(d1[str(dims)+ d3]['train_loss'+ d3], label = " train loss" + str(dims), markersize=10,  linewidth=3)
            plt.plot(d1[str(dims) + d3]['valid_loss' + d3], label = " validation loss" + str(dims), markersize=10,  linewidth=3, linestyle='dashed')

        plt.legend(loc = 'upper left',  prop = {'size': 15})
    
        results_filename = 'PyTorch_MLP_Loss_with_Batch_Norm_' +d3 + '.png'
        results_filename = os.path.join(path, results_filename)
        plt.savefig(results_filename,bbox_inches='tight')
        plt.clf()
        
    for use in batch_norm_use:
        path = './'
        d3 = str(use)
        
        plt.title('PyTorch_MLP_Accuracy_with_Batch_Norm = '+ d3)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
    
    
        for dims in hidden_dims_list:
            plt.plot(d1[str(dims)+ d3]['train_accuracy'+ d3], label = "train accuracy for " + str(dims) , markersize=10,  linewidth=3)
            plt.plot(d1[str(dims) + d3]['validation_accuracy' + d3], label = " validation accuracy for " + str(dims), markersize=10,  linewidth=3, linestyle='dashed')

        plt.legend(loc = 'upper left',  prop = {'size': 15})
    
        results_filename = 'PyTorch_MLP_Accuracy_with_Batch_Norm_' +d3 + '.png'
        results_filename = os.path.join(path, results_filename)
        plt.savefig(results_filename,bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.txt' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)