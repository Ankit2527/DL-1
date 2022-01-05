# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:12:38 2021

@author: Ankit
"""

###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt

from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set
import copy 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
from tqdm import tqdm
import glob

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*3, num_classes)
            )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
            cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model

def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #Load the datasets
    checkpoint_name = os.path.join('./CNN/' + checkpoint_name)
    tfb_name_train = os.path.join(checkpoint_name + '/train')
    tfb_name_validation  = os.path.join(checkpoint_name + '/valid')
    if not os.path.exists(tfb_name_train):
        os.makedirs(tfb_name_train)
    if not os.path.exists(tfb_name_validation):
        os.makedirs(tfb_name_validation)

    writer = SummaryWriter(tfb_name_train)
    validation_writer = SummaryWriter(tfb_name_validation)
    
    save_graph = 'True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #Loading the dataset
    train_set, val_set = get_train_validation_set(data_dir,validation_size = 5000)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,
                                          shuffle = True, num_workers = 2, pin_memory = True)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle = True, num_workers = 2, pin_memory = True)

    #Initialize the optimizers and learning rate scheduler. 
    #We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [90, 135], gamma = 0.1)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    train_loss = []
    train_accuracy = []
    valid_loss = []
    validation_accuracy =[]
    total_train_labels = 0
    total_validation_labels = 0
    train_correct = 0
    validation_correct = 0
    d1 = {} 
    min_valid_loss = np.inf
    path = checkpoint_name
    fname = checkpoint_name + 'model.pth'
    best_fname = 'model_best.pth'
    fname = os.path.join(path, fname)
    best_fname = os.path.join(path, best_fname)
    if save_graph:
      if not(os.path.exists(path)):
        os.makedirs(path)

    for epoch in range(epochs):
        running_loss = 0
        validation_running_loss = 0
        #Training
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss +=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            _, predicted = outputs.max(1)
            total_train_labels += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        scheduler.step()
        print('Epoch', int(epoch+1))
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * train_correct / total_train_labels
        
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        
        print(f'epoch [{epoch + 1}/{epochs}], Train loss:{epoch_loss:.4f}')
        print('Accuracy of the network on the train images:', 100 * train_correct / total_train_labels)
        
        writer.add_scalar('Training Loss',epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)

        #Validation
        model.eval()
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                validation_outputs = model(inputs)
                v_loss = criterion(validation_outputs, labels)
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
            
            validation_writer.add_scalar('Validation Loss', validation_epoch_loss, epoch )
            validation_writer.add_scalar('Validation Accuracy', validation_epoch_accuracy, epoch)
        
        if min_valid_loss > validation_epoch_loss:
            min_valid_loss = validation_epoch_loss
            model_best = copy.deepcopy(model)
            if save_graph:
              torch.save(model.state_dict(), best_fname)
            
    d1['train_loss']= train_loss
    d1['train_accuracy'] = train_accuracy
    d1['validation_loss'] = valid_loss
    d1['validation_accuracy'] = validation_accuracy
    
    results_filename = "Saved_values_for_accuracy_and_loss.json"
    json_filename = os.path.join(checkpoint_name, results_filename)

    with open(json_filename, 'w') as f:
        json.dump(d1, f, indent = 4)
    
    print('Finished Training')

    return model

def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    model = model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss_epoch = 0
    test_correct = 0
    total_test_labels = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            test_outputs = model(inputs)
            test_loss = criterion(test_outputs, labels)
            test_loss_epoch += test_loss.item()
            
            _, predicted = test_outputs.max(1)
            total_test_labels += labels.size(0)
            test_correct += (predicted == labels).sum().item()

        test_epoch_loss = test_loss_epoch / len(data_loader)
        no_augmentation = 100 * test_correct / total_test_labels
        
    return test_accuracy_no_augmentation

def test_model(model, batch_size, data_dir, checkpoint_name, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    set_seed(seed)
    test_results_plain = {}
    test_results = {}
    transform_test_set = [gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform]
    
    model_name = checkpoint_name
    checkpoint_folder = os.path.join('./CNN/' + checkpoint_name)
    best_fname = 'model_best.pth'
    best_fname = os.path.join(checkpoint_folder, best_fname)
    state_dict = torch.load(best_fname)
    model.load_state_dict(state_dict)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    #Test
    with torch.no_grad():
        for transform in transform_test_set:
            test_accuracies_plain = []
            test_accuracies = []
            for i in range(0, 6):
                t_set = transform(severity=i)
                if (i ==0):
                    test_set = get_test_set(data_dir, augmentation=None)
                    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 2)
                else:
                    test_set = get_test_set(data_dir, augmentation=t_set)
                    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 2)
                    
                test_accuracy = evaluate_model(model, test_loader, device)
                test_error = 100 - test_accuracy
                
                dummy = transform.__name__
                if (i == 0):
                    test_results_plain[str(dummy) +'_'+ str(i)] = test_error
                else:
                    test_results[str(dummy) +'_'+ str(i)] = test_error
            test_accuracies.append(test_error)
    
    json_file = os.path.join('./', model_name + '_no_augmentation' +'.json')
    with open(json_file, 'w') as f:
        json.dump(test_results_plain, f, indent = 4)
    
    json_file = os.path.join('./CNN', model_name +'.json')
    with open(json_file, 'w') as f:
        json.dump(test_results, f, indent = 4)

    return test_results

def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(seed)
    model = get_model(model_name, num_classes = 10)
    checkpoint_name = model_name
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device)
    test = test_model(model, batch_size, data_dir, checkpoint_name, device, seed)

if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--model_name', default='densenet121', type=str,
                        help='Name of the model to train.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)