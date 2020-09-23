"""
Testing RNNVAE for a single channel, with MNIST dataset

Training, visualization and validation.

Most of the code taken from
https://github.com/crazysal/VariationalRNN/blob/master/VariationalRecurrentNeuralNetwork-master/train.py 
"""

import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from rnnvae.rnnvae import ModelRNNVAE

#hyperparameters
x_size = 28
h_size = 100
z_dim = 16
hidden = 16
n_layers =  1
n_epochs = 10
clip = 10
learning_rate = 1e-3
batch_size = 128
seed = 128

#manual seed
torch.manual_seed(seed)
plt.ion()

dataset_train = datasets.MNIST('data', train=True, download=True,
		                       transform=transforms.ToTensor())


dataset_test = datasets.MNIST('data', train=False, 
		                      transform=transforms.ToTensor())

#init model + optimizer + datasets
train_loader = torch.utils.data.DataLoader(dataset_train,
               batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset_test, 
              batch_size=batch_size, shuffle=True)

# Define model and optimizer
model = ModelRNNVAE(x_size, h_size, hidden, n_layers, 
                     hidden, n_layers, hidden,
                     n_layers, z_dim, hidden, n_layers,
                     clip)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.init_loss()
model.optimizer = optimizer
# Fit the model
model.fit(batch_size, train_loader)

# Evaluate with the test set
#NOT YET IMPLEMENTED
#model.evaluate(test_loader)