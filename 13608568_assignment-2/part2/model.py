# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 18:46:06 2021

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

import math
import torch
import torch.nn as nn



class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """
    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.W_gx = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_g =  nn.Parameter(torch.Tensor(lstm_hidden_dim))

        self.W_ix = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_i =  nn.Parameter(torch.Tensor(lstm_hidden_dim))

        self.W_fx = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_f =  nn.Parameter(torch.Tensor(lstm_hidden_dim))

        self.W_ox = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_o =  nn.Parameter(torch.Tensor(lstm_hidden_dim))
        
        self.prev_c = torch.zeros((self.hidden_dim)).to(device)
        self.prev_h = torch.zeros((self.hidden_dim)).to(device)
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for name, param in self.named_parameters():
            if str(name) == 'b_f':
                param.data.uniform_(-stdv , stdv) + 1
            else:
                param.data.uniform_(-stdv, stdv)


    def forward(self, embeds, h_0, c_0):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        
        #h_0 and c_0 are the previous state hidden states
        self.prev_c = c_0
        self.prev_h = h_0
        
        
        inp_len, _, _ = embeds.size()
        seq = []

        #Going through each time step (teacher forcing)
        for t in range(inp_len):
            x = embeds[t, :, :]
            
            g = torch.tanh(x @ self.W_gx + self.prev_h @ self.W_gh  + self.b_g)
            i = torch.sigmoid(x @ self.W_ix + self.prev_h @ self.W_ih + self.b_i)
            f = torch.sigmoid(x @ self.W_fx + self.prev_h @ self.W_fh + self.b_f)
            o = torch.sigmoid(x @ self.W_ox + self.prev_h @ self.W_oh + self.b_o)
            c = g * i + self.prev_c * f
            h = torch.tanh(c) * o
            self.prev_c = c
            self.prev_h = h
            
            seq.append(self.prev_h.unsqueeze(0))
        
        seq = torch.cat(seq, dim=0)
        return seq, self.prev_h, self.prev_c


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """
    def __init__(self, lstm_hidden_dim, embedding_size, device, vocabulary_size):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding_size = embedding_size
        self.device = device 
        
        self.embed = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = LSTM(lstm_hidden_dim, embedding_size)
        self.linear = nn.Linear(lstm_hidden_dim, vocabulary_size)


    def forward(self, x, h_0, c_0):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        out_ = self.embed(x)
        out_, h_n, c_n = self.lstm(out_, h_0, c_0)
        out_ = self.linear(out_)
        
        return out_

    def sample(self, batch_size = 4, sample_length = 30, temperature = 0):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        self.batch_size = batch_size
        self.sample_length = sample_length
        self.temperature = temperature
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Initialize the hidden states to zero for the sample generation
        h_inp = torch.zeros(self.lstm_hidden_dim).to(device)
        c_inp = torch.zeros(self.lstm_hidden_dim).to(device)
        
        #Setting a random initial character
        out = torch.empty(sample_length, batch_size).to(device)
        out[0] = torch.randint(1, self.vocabulary_size, (batch_size,)).to(device)

        for i in range(1, sample_length):
            
            x = self.embed(out[i-1].long())

            if i == 1:
                x, h_t, c_t = self.lstm(x.unsqueeze(0), h_inp, c_inp)
            else:
                h_0 = h_new
                c_0 = c_new
                x, h_t, c_t = self.lstm(x.unsqueeze(0), h_0, c_0)
            
            x = self.linear(x.squeeze())
            
            if self.temperature == 0:
                out[i] = torch.argmax(x, dim = -1)
                
            else:
                preds = torch.softmax(self.temperature * x, dim = -1)
                distribution = torch.distributions.Categorical(preds)
                out[i] = distribution.sample((1, )).squeeze()
            
            #Update the hidden states for the following time steps
            h_new = h_t
            c_new = c_t
        
        return out.T.int()
