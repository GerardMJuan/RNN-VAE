"""
Test the RNNVAE with real MRI longitudinal data.
"""

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
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss

#hyperparameters
x_size = 40
h_size = 20
z_dim = 5
hidden = 5
n_layers = 1
n_epochs = 100
clip = 10
learning_rate = 1e-3
batch_size = 128
seed = 1714

# out_dir
out_dir = "experiments/base_test/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# LOAD DATA
csv_path = "data/tadpole_mrionly.csv"
X_train, X_test = open_MRI_data_var(csv_path, train_set=0.8, normalize=True)
#List of (nt, nfeatures) numpy objects

nfeatures = X_train[0].shape[1]

# Apply padding to both X_train and X_val
X_train_tensor = [ torch.FloatTensor(t) for t in X_train ]
X_train_pad = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False)
X_test_tensor = [ torch.FloatTensor(t) for t in X_test ]
X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False)

# Those datasets are of size [Tmax, Batch_size, nfeatures]

# Save mask to unpad later when testing
mask_train = X_train_pad > 0
mask_test = X_test_pad > 0

#Create the dataloaders
# train_loader = torch.utils.data.DataLoader(dataset_train,
#                batch_size=batch_size, shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset_test, 
#               batch_size=batch_size, shuffle=True)

#TODO: DO WE NEED TO NORMALIZE SOME THINGS?

# Define model and optimizer
model = ModelRNNVAE(x_size, h_size, hidden, n_layers, 
                     hidden, n_layers, hidden,
                     n_layers, z_dim, hidden, n_layers,
                     clip, n_epochs, batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.optimizer = optimizer

# Fit the model
# model.fit(train_loader, test_loader)
model.fit(X_train_pad, X_test_pad)

### After training, save the model!
model.save(out_dir, 'model.pt')

# Predict the reconstructions from X_val and X_train
X_test_fwd = model.predict(X_test_pad)
X_train_fwd = model.predict(X_train_pad)

X_test_hat = X_test_fwd["xnext"]
X_train_hat = X_train_fwd["xnext"]

# Unpad using the masks
#after masking, need to rehsape to (nt, nfeat)
X_test_hat = [X[mask_test[:,i,:]].reshape((-1, nfeatures)) for (i, X) in enumerate(X_test_hat)]
X_train_hat = [X[mask_train[:,i,:]].reshape((-1, nfeatures)) for (i, X) in enumerate(X_train_hat)]

#Compute mean absolute error over all sequences
mse_train = np.mean([mean_absolute_error(xval, xhat) for (xval, xhat) in zip(X_train, X_train_hat)])
print('MSE over the train set: ' + str(mse_train))

#Compute mean absolute error over all sequences
mse_test = np.mean([mean_absolute_error(xval, xhat) for (xval, xhat) in zip(X_test, X_test_hat)])
print('MSE over the test set: ' + str(mse_test))

#plot validation and 
plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir, "total_loss.png")
plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir, "kl_loss.png")
plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir, "ll_loss.png") #Negative to see downard curve

# Visualization of trajectories
subj = 6
feature = 12
# For train
plot_trajectory(X_train, X_train_hat, subj, 'all', out_dir, f'traj_train_s_{subj}_f_all') # testing for a given subject
plot_trajectory(X_train, X_train_hat, subj, feature, out_dir, f'traj_train_s_{subj}_f_{feature}') # testing for a given feature

# For test
plot_trajectory(X_test, X_test_hat, subj, 'all', out_dir, f'traj_test_s_{subj}_f_all') # testing for a given subject
plot_trajectory(X_test, X_test_hat, subj, feature, out_dir, f'traj_test_s_{subj}_f_{feature}') # testing for a given feature

# get latent vectors
import ipdb; ipdb.set_trace()
z_train = np.array([X_train_fwd['zx'][i].detach().numpy() for i in range(len(X_train_fwd['zx']))])
z_test = np.array([X_test_fwd['zx'][i].detach().numpy() for i in range(len(X_train_fwd['zx']))])

# Visualize latent space

# plot a 2D space between 2 specified latent dimension for a specific time

# plot trajectory for a single latent dimension over time, for specific subject

# plot trajectory for 2D space over two latent dimensions, with coloring for timepoints.

#Do many sampling from the latent space

#Visualized the sampled values using same methods as before