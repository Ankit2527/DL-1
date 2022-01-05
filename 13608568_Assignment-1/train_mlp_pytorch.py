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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import copy


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    batch_size = len(predictions)
    preds = np.argmax(predictions, axis =1)
    correct = np.sum(preds == targets.reshape(-1))
    accuracy = correct / float(batch_size)
    
    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loss_epoch = 0
    test_correct = 0
    total_test_labels = 0
    loss_module =  nn.CrossEntropyLoss()

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.reshape(inputs.shape[0], -1)
        
        #run pass
        test_outputs = model(inputs)
        test_loss = loss_module(test_outputs, labels)
        test_loss_epoch += test_loss.item()

        _, predicted = test_outputs.max(1)
        total_test_labels += labels.size(0)
        test_correct += (predicted == labels).sum().item()

    test_epoch_loss = test_loss_epoch / len(data_loader)
    avg_accuracy = 100 * test_correct / total_test_labels    


    
    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir, save_graph=True):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)



    train_loader = cifar10_loader['train']
    validation_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']


    n_classes = 10
    model = MLP(3* 32 * 32, hidden_dims, n_classes, use_batch_norm)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1)
    model.to(device)
    
    print('Model Summary\n', model)
    print(device)
  
    train_loss=[]
    valid_loss = []
    train_accuracy = []
    validation_accuracy =[]
    total_train_labels = 0
    total_validation_labels = 0
    train_correct =0
    validation_correct = 0 
    min_valid_loss = np.inf
    path = './'
    fname = 'model_mlp_pytorch.pth'
    best_fname = 'best_' + fname
    fname = os.path.join(path, fname)
    best_fname = os.path.join(path, best_fname)
    if save_graph:
      if not(os.path.exists(path)):
        os.makedirs(path)
      

    for epoch in range(epochs):  
        running_loss = 0
        validation_running_loss = 0
        model.train()
        
        #Training
        for inputs, labels in train_loader:
            inputs=inputs.to(device)
            print('Inputs',inputs.shape)
            labels = labels.to(device)
            print('Labels',labels.shape)
            inputs = inputs.reshape(inputs.shape[0], -1)
            print('Transformed Inputs',inputs.shape)
            model.zero_grad()
            outputs = model(inputs)
            print('Outputs', outputs.shape)
            loss = loss_module(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, predicted = outputs.max(1)
            total_train_labels += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * train_correct / total_train_labels

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        
        print(f'epoch [{epoch + 1}/{epochs}], Train loss:{epoch_loss:.4f}')
        print('Accuracy of the network on the train images:', 100 * train_correct / total_train_labels)
        

        #Validation
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.reshape(inputs.shape[0], -1)
            validation_outputs = model(inputs)
            v_loss = loss_module(validation_outputs, labels)
            validation_running_loss += v_loss.item()

            _, predicted = validation_outputs.max(1)
            total_validation_labels += labels.size(0)
            validation_correct += (predicted == labels).sum().item()

        validation_epoch_loss = validation_running_loss / len(validation_loader)
        validation_epoch_accuracy = 100 * validation_correct / total_validation_labels
        
        valid_loss.append(validation_epoch_loss)
        validation_accuracy.append(validation_epoch_accuracy)
        
        print(f'epoch [{epoch + 1}/{epochs}], Validation loss:{validation_epoch_loss:.4f}')
        print('Accuracy of the network on the validation images: ', validation_epoch_accuracy)
        
        if min_valid_loss > validation_epoch_loss:
            min_valid_loss = validation_epoch_loss
            optimized_model = copy.deepcopy(model)
            if save_graph:
                #print("Saving model", best_fname)
                torch.save(model.state_dict(), best_fname)
            
    if save_graph:
        #Loss plot for train and validation
        tl, = plt.plot(train_loss, label='Pytorch MLP Training Loss')   
        vl, = plt.plot(valid_loss, label='Pytorch MLP Validation Loss')   
        plt.legend(loc = 'upper right',  prop = {'size': 20})
        plt.legend(handles = [tl, vl])
        plt.title('Loss_Pytorch_MLP')
        plt.xlabel('Epochs')
        fname = 'Loss_Pytorch_MLP'  + '.png'
        fname = os.path.join(path, fname)
        plt.savefig(fname,bbox_inches='tight')
        plt.show()    
        
        
        #Accuracy plot for train and validation
        ta, = plt.plot(train_accuracy, label='Pytorch MLP Training Accuracy')   
        va, = plt.plot(validation_accuracy, label='Pytorch MLP Validation Accuracy')  
        plt.legend(loc = 'upper right',  prop = {'size': 20})
        plt.legend(handles = [ta, va])
        plt.title('Accuracy_Pytorch_MLP')
        plt.xlabel('Epochs')
        fname = 'Accuracy_Pytorch_MLP'  + '.png'
        fname = os.path.join(path, fname)
        plt.savefig(fname,bbox_inches='tight')
        plt.show()  
    
    print('Finished Training')
    test_accuracy = evaluate_model(optimized_model, test_loader)   
    print("Accuracy on test set with :", hidden_dims, '=', test_accuracy)

    return model, train_loss, valid_loss, train_accuracy, validation_accuracy


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    