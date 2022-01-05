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
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import *
#from networks import *
from networks import *
import os
from time import gmtime, strftime

import copy 
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


def permute_indices(molecules: Batch) -> Batch:
    """permute the atoms within a molecule, but not across molecules

    Args:
        molecules: batch of molecules from pytorch geometric

    Returns:
        permuted molecules
    """
    # Permute the node indices within a molecule, but not across them.
    ranges = [
        (i, j) for i, j in zip(molecules.ptr.tolist(), molecules.ptr[1:].tolist())
    ]
    permu = torch.cat([torch.arange(i, j)[torch.randperm(j - i)] for i, j in ranges])

    n_nodes = molecules.x.size(0)
    inits = torch.arange(n_nodes)
    # For the edge_index to work, this must be an inverse permutation map.
    translation = {k: v for k, v in zip(permu.tolist(), inits.tolist())}

    permuted = deepcopy(molecules)
    permuted.x = permuted.x[permu]
    # Below is the identity transform, by construction of our permutation.
    permuted.batch = permuted.batch[permu]
    permuted.edge_index = (
        permuted.edge_index.cpu()
        .apply_(translation.get)
        .to(molecules.edge_index.device)
    )
    return permuted


def compute_loss(
    model: nn.Module, molecules: Batch, criterion: Callable
) -> torch.Tensor:
    """use the model to predict the target determined by molecules. loss computed by criterion.

    Args:
        model: trainable network
        molecules: batch of molecules from pytorch geometric
        criterion: callable which takes a prediction and the ground truth 

    Returns:
        loss

    TODO: 
    - conditionally compute loss based on model type
    - make sure there are no warnings / errors
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = get_labels(molecules)
    labels = labels.to(device)
    labels = labels.unsqueeze(1)
    
    #check model name and labels
    if 'mlp' in str(type(model)).lower():
        inputs = get_mlp_features(molecules)
        inputs = inputs.to(device)
        outputs = model(inputs)

    if 'gnn' in str(type(model)).lower():
        molecules = molecules.to(device)
        molecules['x'] = molecules['x'][:, :5]
        outputs = model(molecules['x'], molecules['edge_index'], molecules['edge_attr'], molecules['batch'])

    loss = criterion(outputs, labels)
    return loss


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: Callable, permute: bool
) -> float:
    """
    Performs the evaluation of the model on a given dataset.

    Args:
        model: trainable network
        data_loader: The data loader of the dataset to evaluate.
        criterion: loss module, i.e. torch.nn.MSELoss()
        permute: whether to permute the atoms within a molecule
    Returns:
        avg_loss: scalar float, the average loss of the model on the dataset.

    Hint: make sure to return the average loss of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    
    TODO: conditionally permute indices
          calculate loss
          average loss independent of batch sizes
          make sure the model is in the correct mode
    """
    # # check if permute is true or flase is true
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    for inputs in data_loader:
        running_loss = 0
        if permute:
            inputs = permute_indices(inputs)
        with torch.no_grad():
            loss = compute_loss(model, inputs, criterion)
            running_loss +=loss.item()
            
    avg_loss = round(running_loss / len(data_loader), 20)
    return avg_loss


def train(
    model: nn.Module, lr: float, batch_size: int, epochs: int, seed: int, data_dir: str
):
    """a full training cycle of an mlp / gnn on qm9.

    Args:
        model: a differentiable pytorch module which estimates the U0 quantity
        lr: learning rate of optimizer
        batch_size: batch size of molecules
        epochs: number of epochs to optimize over
        seed: random seed
        data_dir: where to place the qm9 data

    Returns:
        model: the trained model which performed best on the validation set
        test_loss: the loss over the test set
        permuted_test_loss: the loss over the test set where atomic indices have been permuted
        val_losses: the losses over the validation set at every epoch
        logging_info: general object with information for making plots or whatever you'd like to do with it

    TODO:
    - Implement the training of both the mlp and the gnn in the same function
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

    # Loading the dataset
    #check model name 
    model_name = args.model
    train, valid, test = get_qm9(data_dir, model.device)
    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        exclude_keys=["pos", "idx", "z", "name"],
    )
    validation_dataloader = DataLoader(
        valid, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    test_dataloader = DataLoader(
        test, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO: Initialize loss module and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [90, 135], gamma = 0.1)
    
    # TODO: Training loop including validation, using evaluate_model
    train_loss = []
    valid_loss = []

    min_valid_loss = np.inf
    path = model_name
    filename = 'model_best.pth'
    filename = os.path.join(path, filename)
    save_graph = 'True'
    if save_graph:
      if not(os.path.exists(path)):
        os.makedirs(path)
    print('Model', model)

    for epoch in range(epochs):  
        running_loss = 0
        validation_running_loss = 0
        model.train()
        for i, (inputs) in enumerate(train_dataloader):

            inputs = inputs.to(device)
            loss = compute_loss(model, inputs, criterion)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
        epoch_loss = round(running_loss / len(train_dataloader), 20)
        train_loss.append(epoch_loss)
        print(f'epoch [{epoch + 1}/{epochs}], Train loss:{epoch_loss:.4f}')
    

        with torch.no_grad():
            model.eval()
            for inputs in validation_dataloader:
                inputs = inputs.to(device)
                validation_loss = compute_loss(model, inputs, criterion)
                validation_running_loss += validation_loss.item()

            validation_epoch_loss = round(validation_running_loss / len(validation_dataloader), 20)
            valid_loss.append(validation_epoch_loss)
            print(f'epoch [{epoch + 1}/{epochs}], Validation loss:{validation_epoch_loss:.4f}')
        
        
        if min_valid_loss > validation_epoch_loss:
            min_valid_loss = validation_epoch_loss
            model_best = copy.deepcopy(model)
            if save_graph:
              torch.save(model.state_dict(), filename)
            
    print('Finished Training')

    # # TODO: Test best model
    test_loss = evaluate_model(model, test_dataloader, criterion, permute = False)
    # # TODO: Test best model against permuted indices
    permuted_test_loss = evaluate_model(model, test_dataloader, criterion, permute = True)
    # # TODO: Add any information you might want to save for plotting
    logging_info = train_loss
    
    print('Test loss  = ', test_loss)
    print('Permuted Test Loss = ', permuted_test_loss)

    return model_best, test_loss, permuted_test_loss, valid_loss, logging_info


def main(**kwargs):
    """main handles the arguments, instantiates the correct model, tracks the results, and saves them."""
    which_model = kwargs.pop("model")
    mlp_hidden_dims = kwargs.pop("mlp_hidden_dims")
    gnn_hidden_dims = kwargs.pop("gnn_hidden_dims")
    gnn_num_blocks = kwargs.pop("gnn_num_blocks")

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if which_model == "mlp":
        model = MLP(FLAT_INPUT_DIM, mlp_hidden_dims, 1)
    elif which_model == "gnn":
        model = GNN(
            n_node_features=Z_ONE_HOT_DIM,
            n_edge_features=EDGE_ATTR_DIM,
            n_hidden=gnn_hidden_dims,
            n_output=1,
            num_convolution_blocks=gnn_num_blocks,
        )
    else:
        raise NotImplementedError("only mlp and gnn are possible models.")

    model.to(device)
    model, test_loss, permuted_test_loss, validation_loss, logging_info = train(model, **kwargs)

    #Loss Plot for Validation        
    vl, = plt.plot(validation_loss, label = 'Validation Loss')   
    plt.legend(loc = 'upper right',  prop = {'size': 20})
    plt.legend(handles = [vl])
    plt.title('Validation_Loss')
    plt.xlabel('Epochs')
    fname = 'Loss'  + '.png'
    path = "./gnn/"
    fname = os.path.join(path, fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        default="gnn",
        type=str,
        choices=["mlp", "gnn"],
        help="Select between training an mlp or a gnn.",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        default=[128, 128, 128, 128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the mlp. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--gnn_hidden_dims",
        default=64,
        type=int,
        help="Hidden dimensionalities to use inside the mlp. The same number of hidden features are used at every layer.",
    )
    parser.add_argument(
        "--gnn_num_blocks",
        default=2,
        type=int,
        help="Number of blocks of GNN convolutions. A block may include multiple different kinds of convolutions (see GNN comments)!",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=5, type=int, help="Max number of epochs")

    # Technical
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the qm9 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
