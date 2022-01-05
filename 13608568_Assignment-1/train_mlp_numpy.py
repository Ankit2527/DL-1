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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
import copy
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
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
    
    test_loss_epoch = 0
    test_correct_batch = 0
    loss_module = CrossEntropyModule()
    
    for inputs, labels in data_loader:
        inputs = inputs.reshape(inputs.shape[0], -1)
        labels = labels.reshape(-1, 1)
        test_pred = model.forward(inputs)
        C = np.max(labels) +1 
        labels_one_hot =one_hot(labels, C)
        if (labels_one_hot.shape == test_pred.shape):
            test_loss = loss_module.forward(test_pred, labels)
            test_loss_epoch += test_loss
            test_correct_batch += accuracy(test_pred, labels)
        else:
            pass
    test_epoch_loss = test_loss_epoch / len(data_loader)
    avg_accuracy = 100 * test_correct_batch / len(data_loader)
    
    #print("Test Loss : ", test_epoch_loss)
    #print("Test Accuracy: ", avg_accuracy)

    return avg_accuracy


def one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    dummy = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[dummy + labels.ravel()] = 1
    return labels_one_hot

def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)
    
    train_loader = cifar10_loader['train']
    validation_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']
    
    n_classes = 10
    model = MLP(3* 32 * 32, hidden_dims, n_classes)
    loss_module = CrossEntropyModule()
    
    train_loss = []
    valid_loss = []
    train_accuracy = []
    validation_accuracy = []
    total_train_labels = 0
    
    validation_epoch_loss = 0
    min_valid_loss = np.inf
    
    path = './'
    filename = 'model_mlp_numpy.pth'
    fname = filename
    best_fname = 'best_' + filename
    fname = os.path.join(path, fname)
    best_fname = os.path.join(path, best_fname)
    if not(os.path.exists(path)):
      os.makedirs(path)
    
    for epoch in range(epochs):
        running_loss = 0
        validation_running_loss = 0
        validation_correct = 0
        train_correct = 0
        
        
        #Training
        for inputs, labels in train_loader:
            #Forward
            inputs = inputs.reshape(inputs.shape[0], -1)
            labels = labels.reshape(-1, 1)
            outputs = model.forward(inputs)
            loss = loss_module.forward(outputs, labels)
            
            #Backward
            backprop_loss = loss_module.backward(outputs, labels)
            model.backward(backprop_loss)
            
            #Accumulate loss
            running_loss += loss
            
            for x in range(0, len(model.layers), 2):
                ll = model.layers[x]
                ll.params['weight'] -= lr * ll.grads['weight'] 
                ll.params['biases'] -= lr * ll.grads['biases'] 
            
            predicted = outputs.max(1)
            total_train_labels += len(labels)
            train_correct += accuracy(outputs, labels)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * train_correct / len(train_loader)
        
        train_loss.append(epoch_loss)  
        train_accuracy.append(epoch_accuracy) 
            
        print(f'epoch [{epoch + 1}/{epochs}], Train loss:{epoch_loss:.4f}')
        print('Accuracy of the network on the train images:', epoch_accuracy)
        
        
        for inputs, labels in validation_loader:
            inputs = inputs.reshape(inputs.shape[0], -1)
            labels = labels.reshape(-1, 1)
            validation_outputs = model.forward(inputs)
            C = np.max(labels) +1 
            labels_one_hot = one_hot(labels, C)
            if (labels_one_hot.shape == validation_outputs.shape):
                validation_loss = loss_module.forward(validation_outputs, labels)
                validation_running_loss += validation_loss
                validation_correct += accuracy(validation_outputs, labels)
            else:
                pass
        validation_epoch_loss = validation_running_loss / len(validation_loader)
        validation_epoch_accuracy = 100 * validation_correct / len(validation_loader)
        
        valid_loss.append(validation_epoch_loss)  
        validation_accuracy.append(validation_epoch_accuracy) 
        
        print(f'epoch [{epoch + 1}/{epochs}], Validation loss:{validation_epoch_loss:.4f}')
        print('Accuracy of the network on the validation images: ', validation_epoch_accuracy)
        
        if min_valid_loss > validation_epoch_loss:
             min_valid_loss = validation_epoch_loss
             best_model = copy.deepcopy(model)
    
    print('Finished Training')
    
    
    #Loss plot for train and validation
    tl, = plt.plot(train_loss, label='Numpy MLP Training Loss')   
    vl, = plt.plot(valid_loss, label='Numpy MLP Validation Loss')   
    plt.title('Loss_Numpy_MLP')
    plt.xlabel('Epochs')
    plt.legend(loc = 'upper right',  prop = {'size': 20})
    plt.legend(handles = [tl, vl])
    fname = 'Loss_Numpy_MLP'  + '.png'
    fname = os.path.join(path, fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.show()    
    
    
    #Accuracy plot for train and validation
    ta, = plt.plot(train_accuracy, label='Numpy MLP Training Accuracy')   
    va, = plt.plot(validation_accuracy, label='Numpy MLP Validation Accuracy')   
    plt.title('Accuracy_Numpy_MLP')
    plt.xlabel('Epochs')
    plt.legend(loc = 'upper right',  prop = {'size': 20})
    plt.legend(handles = [ta, va])
    fname = 'Accuracy_Numpy_MLP'  + '.png'
    fname = os.path.join(path, fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.show()  
    
    test_accuracy = evaluate_model(best_model, test_loader)
    print("Accuracy on test set : ",  test_accuracy)

    return best_model, validation_accuracy, test_accuracy


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
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
    