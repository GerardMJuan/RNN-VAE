"""
File containing the model of the network.

This network is divided in three parts:

Encoder, which from an input x_t and a previous hidden state h_t-1
generates a mu and sigma that generate a latent space z_t.

Decoder, which from a sampled z_t and a hidden state h_t-1 generates
x_hat_t

RNN, that samples from z_t and with x_t and h_t-1 gives a new hidden
state h_t, for the next input.

This network is multi channel, that is, it takes as inputs different channels of data
and jointly combines them

The three parts of the network are combined in the class model_MCRNNVAE


Alternative implmeentation:
https://github.com/crazysal/VariationalRNN/blob/master/VariationalRecurrentNeuralNetwork-master/model.py

"""

import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence
import os


