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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
import numpy as np

class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs):
        """
        Initializes MLP object.
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_outputs: This number is required in order to specify the
                     output dimensions of the MLP
        TODO: 
        - define a simple MLP that operates on properly formatted QM9 data
        """
        super().__init__()
        network = []
        num_mlp = [n_inputs] + n_hidden
        for i in range(1, len(num_mlp)):
            network += [nn.Linear(num_mlp[i-1], num_mlp[i]),
                        nn.ReLU()]
      
        network += [nn.Linear(num_mlp[-1], n_outputs)]
        self.layers = nn.Sequential(*network)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
            x: input to the network
        Returns:
            out: outputs of the network
        """

        out = x
        for mods in self.layers:
          out = mods(out)


        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device



class GNN(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden: int,
        n_output: int,
        num_convolution_blocks: int,
    ) -> None:
        """create the gnn

        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural networks (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. A block may include multiple convolutions
        
        TODO: 
        - define a GNN which has the following structure: node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        - One the data has been pooled, it may be beneficial to apply another MLP on the pooled data before predicing the output.
        - Do we need an MLP added to the GNN? What about node embedding in the model layers. 
        
        """
        super().__init__()
        num_input_channels, out_channels_2 = n_node_features, n_hidden
        self.embedding = nn.Sequential(nn.Linear(num_input_channels, n_hidden))
        self.gnn_layers = []
        for i in range(1,(num_convolution_blocks + 1)):
          self.gnn_layers += [nn.ReLU(inplace=True), 
                        geom_nn.RGCNConv(in_channels = n_hidden, 
                                        out_channels = n_hidden, 
                                        num_relations = 4),
                        nn.ReLU(inplace = True), 
                        geom_nn.MFConv(in_channels = n_hidden, 
                                       out_channels = n_hidden,)
                        ]
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.head = nn.Sequential(
                      nn.Linear(n_hidden, n_hidden), 
                      nn.ReLU(inplace = True), 
                      nn.Linear(n_hidden, n_output))
        

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node

        Returns:
            prediction

        TODO: implement the forward pass being careful to apply MLPs only where they are allowed!

        Hint: remember to use global pooling. [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        """
        edge_attr = edge_attr.argmax(-1)

        x = self.embedding(x)
        for i, layer in enumerate(self.gnn_layers):
            if 'RGCN' in str(layer):
                x = layer(x, edge_index, edge_attr)
            elif 'MFC' in str(layer):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        x = geom_nn.global_add_pool(x, batch)
        x = self.head(x)

        return x

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
