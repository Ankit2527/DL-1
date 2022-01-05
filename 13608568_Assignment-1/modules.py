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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """
        self.in_features = in_features
        self.out_features = out_features
        
        self.params = {'weight' : None, 'biases' : None}
        self.grads = {'weight' : None, 'biases' : None}
        
        w = np.random.randn(self.out_features, self.in_features) * np.sqrt(2.0/self.in_features)
        b = np.zeros((self.out_features))
        
        self.params['weight'] = w
        self.params['biases'] = b
        
        self.grads['weight'] = np.zeros((self.out_features, self.in_features))
        self.grads['biases'] = np.zeros((self.out_features))

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        self.x = x
        out = x @ self.params['weight'].T + self.params['biases'].T

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        self.grads['weight'] = dout.T @ self.x
        self.grads['biases'] = np.ones((self.x.shape[0])).T @ dout
        dx = dout @ self.params['weight']

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        pass


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        self.out = np.maximum(0, x)

        return self.out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        der = np.where(self.out > 0, 1, 0)
        dx = np.multiply(der, dout)

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        pass


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        b = x.max(axis = 1, keepdims = True)
        y = np.exp(x - b)
        self.out = y / y.sum(axis = 1, keepdims = True)

        return self.out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        
        dummy = np.zeros(dout.shape)
        for i in range(0, dout.shape[0]):
          for j in range(0, dout.shape[1]):
            dummy[i, j] = self.out[i, j] * (dout[i, j] - np.sum(dout[i, :]*self.out[i, :]))
            
        dx = dummy

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        pass


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """
        
        def one_hot(labels, num_classes):
            num_labels = labels.shape[0]
            dummy = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[dummy + labels.ravel()] = 1
            return labels_one_hot
        
        C = np.max(y) + 1
        y = one_hot(y, C)
        self.loss = -np.sum(y * np.log(x)) / x.shape[0]

        return self.loss

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """
        
        def one_hot(labels, num_classes):
            num_labels = labels.shape[0]
            dummy = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[dummy + labels.ravel()] = 1
            return labels_one_hot 
    
        C = np.max(y) + 1
        y = one_hot(y, C)
        dx = - (1 / x.shape[0]) * (y / x)

        return dx