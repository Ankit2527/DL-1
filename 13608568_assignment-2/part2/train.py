# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 19:03:33 2021

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
# Date Adapted: 2021-11-11
###############################################################################

from datetime import datetime
import argparse
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel
import time
import matplotlib.pyplot as plt
import os 



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

def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    set_seed(args.seed)
    device = args.device
    print('Using device : ', device)

    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(dataset, args.batch_size, 
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)
    
    # Create model
    model = TextGenerationModel(args.lstm_hidden_dim, 
                                args.embedding_size, 
                                args.device,
                                dataset._vocabulary_size,
                                )
    
    model = model.to(device)
    print('Model Summary\n', model)
    
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    model.to(device)
    
    epochs = args.num_epochs
    total_train_labels = 0
    train_correct = 0
    train_loss = []
    train_accuracy = []
    path = './'
    
    # Training loop
    for epoch in range(epochs):
        i = 0
        running_loss = 0
        train_correct = 0
        model.train()
        
        for i, (inputs, targets) in tqdm(enumerate(data_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            #Initializing the initial hidden states as zeros for LSTM
            h_0 = torch.zeros(args.lstm_hidden_dim).to(device)
            c_0 = torch.zeros(args.lstm_hidden_dim).to(device)
            outputs = model(inputs, h_0, c_0)
        
            loss = loss_module(outputs.view(-1, dataset._vocabulary_size), targets.view(-1))
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_grad_norm)
            optimizer.step()
            
            predictions = torch.argmax(outputs, dim = 2)
            train_correct += (predictions == targets).float().mean().item()
    
        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = train_correct / len(data_loader)
        
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        
        #Print the Loss and Accuracy for the LSTM
        print(f'epoch [{epoch + 1}/{epochs}], Train loss:{epoch_loss:.4f}')
        print('Epoch_Accuracy : ', epoch_accuracy)
        
        
        #Sampling Sentences - change temperate and seed in the arguments  
        #produce different results for temperature and different starting character 
        #to Samples after 1st epoch for sentences of length 30 and 60
        if epoch == 0:
            print('Samples after Epoch 1 \n')
            
            samples = model.sample(4, 30, args.temperature)
            print('Samples of length 30 \n')
            for sample in samples:
                print(dataset.convert_to_string(sample.tolist()))
                
            print('\n')
            samples = model.sample(4, 60, args.temperature)
            print('Samples of length 60 \n')
            for sample in samples:
                print(dataset.convert_to_string(sample.tolist()))
                
        #Samples after 5th epoch for sentences of length 30 and 60        
        if epoch == 4:
            print('Samples after Epoch 5 \n')
            
            samples = model.sample(4, 30, args.temperature)
            print('Samples of length 30 \n')
            for sample in samples:
                print(dataset.convert_to_string(sample.tolist()))
                
            samples = model.sample(4, 60, args.temperature)
            print('Samples of length 60 \n')
            for sample in samples:
                print(dataset.convert_to_string(sample.tolist()))
                
        #Samples at the end of training for sentences of length 30 and 60          
        if epoch == epochs - 1:
            print('Samples at the end of Training \n')
            samples = model.sample(4, 30, args.temperature)
            print('Samples of length 30 \n')
            for sample in samples:
                print(dataset.convert_to_string(sample.tolist()))
                
            samples = model.sample(4, 60, args.temperature)
            print('Samples of length 60 \n')
            for sample in samples:
                print(dataset.convert_to_string(sample.tolist()))
        
    #Loss Plot for Training        
    tl, = plt.plot(train_loss, label='Training Loss')   
    plt.legend(loc = 'upper right',  prop = {'size': 20})
    plt.legend(handles = [tl])
    plt.title('Training_Loss')
    plt.xlabel('Epochs')
    fname = 'LSTM_Loss'  + '.png'
    fname = os.path.join(path, fname)
    plt.savefig(fname,bbox_inches='tight')
    plt.show()    
    
    #Accuracy plot for Training
    ta, = plt.plot(train_accuracy, label='Training Accuracy')   
    plt.legend(loc = 'upper right',  prop = {'size': 20})
    plt.legend(handles = [ta])
    plt.title('Training_Accuracy')
    plt.xlabel('Epochs')
    fname = 'LSTM_Accuracy'  + '.png'
    fname = os.path.join(path, fname)
    plt.savefig(fname,bbox_inches='tight')
    plt.show() 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str,  default = './assets/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')
    parser.add_argument('--temperature', type=float, default=0, choices=["0", "0.5", "1", "2"], help='Temperature for Sample Generation')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, choices=["0", "42", "20", "12", "33"], help='Seed for pseudo-random number generator')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    
    train(args)
