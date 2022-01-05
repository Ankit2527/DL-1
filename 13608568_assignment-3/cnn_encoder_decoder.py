################################################################################
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
# Date Created: 2021-11-27
################################################################################

import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network

        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 FashionMNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size = 3, stride = 2, padding = 1) 
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1) 
        self.conv3 = nn.Conv2d(num_filters, 2 * num_filters, kernel_size = 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size = 3, padding = 1) 
        self.conv5 = nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size = 3, stride = 2, padding = 1)
        self.fc_out = nn.Linear(2 * num_filters * 4 * 4, z_dim)
        self.act_fn = nn.GELU()
        self.flatten = nn.Flatten()
        

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] of type long with values between 0 and 15.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        
        x = x.float() / 15 * 2.0 - 1.0  # Move images between -1 and 1 
        x = self.conv1(x) # 28x28 => 14x14
        x = self.act_fn(x) 
        x = self.conv2(x) # 14x14 => 14x14
        x = self.act_fn(x) 
        x = self.conv3(x) # 14x14 => 7x7
        x = self.act_fn(x)
        x = self.conv4(x) # 7x7 => 7x7
        x = self.act_fn(x)
        x = self.conv5(x) # 7x7 => 4x4
        x = self.act_fn(x)
        x = self.flatten(x)
        mean = self.fc_out(x)
        log_std = self.fc_out(x)
        
        return mean, log_std


class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 16, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.

        Inputs:
            num_input_channels- Number of channels of the image to
                                reconstruct. For a 4-bit FashionMNIST, this parameter is 16
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the decoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.z_dim = z_dim
        
        self.fc_in = nn.Linear(z_dim, 2 * num_filters * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(2 * num_filters, 2 * num_filters, kernel_size = 3, 
                                        output_padding = 0, padding = 1, stride = 2) 
        self.conv2 = nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size = 3, padding = 1) 
        self.conv3 = nn.ConvTranspose2d(2 * num_filters, num_filters, kernel_size = 3, 
                                        output_padding = 1, padding = 1, stride = 2)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1)  
        self.conv5 = nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size = 3, 
                                        output_padding = 1, padding = 1, stride = 2)
        self.act_fn = nn.GELU()
        

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:

            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,num_input_channels,28,28]
        """
        x = self.fc_in(z)
        x = self.act_fn(x)
        x = x.view(-1, 2 * self.num_filters, 4, 4)
        x = self.conv1(x) # 4x4 => 7x7
        x = self.act_fn(x) 
        x = self.conv2(x) # 7x7 => 7x7
        x = self.act_fn(x) 
        x = self.conv3(x) # 7x7 => 14x14
        x = self.act_fn(x)
        x = self.conv4(x) # 14x14 => 14x14
        x = self.act_fn(x)
        x = self.conv5(x) # 14x14 => 28x28
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device

