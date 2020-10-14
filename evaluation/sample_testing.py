# Imports
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import math
import torch
import torch.nn as nn
import numpy as np
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from rnnvae.rnnvae import ModelRNNVAE
from sklearn.metrics import mean_absolute_error
from rnnvae.utils import open_MRI_data_var
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_2d, plot_z_time_2d

import seaborn as sns
import matplotlib.pyplot as plt

# Parameters, and load existing model
#hyperparameters
x_size = 40
h_size = 10
z_dim = 5
hidden = 5
n_layers = 1
n_epochs = 6000
clip = 10
learning_rate = 1e-3
batch_size = 128
seed = 1714
#working dir:

# out_dir
out_dir = "experiments/MRI_padding_test_lin/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model = ModelRNNVAE(x_size, h_size, hidden, n_layers, 
                     hidden, n_layers, hidden,
                     n_layers, z_dim, hidden, n_layers,
                     clip, n_epochs, batch_size)

model.load(out_dir+'model.pt')


#Generate samples
nt = 5
nsamples = 1000
X_sample = model.sample_latent(nsamples, nt)

X_sample['x'] = np.array(X_sample['x']).swapaxes(0,1)
X_sample['z'] = np.array(X_sample['z']).swapaxes(0,1)

# Plot latent space
dim0=1
dim1=0
plot_z_time_2d(X_sample['z'], nt, [dim0, dim1], out_dir, out_name=f'latent_space_{nt}_dim{dim0}_dim{dim1}')
