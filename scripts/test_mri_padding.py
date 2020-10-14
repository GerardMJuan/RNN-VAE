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
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d

def run_experiment(p, csv_path, out_dir):
    """
    Function to run the experiments.
    p contain all the hyperparameters needed to run the experiments
    We assume that all the parameters needed are present in p!!
    out_dir is the out directory
    #hyperparameters
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #Seed
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])

    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    # LOAD DATA
    X_train, X_test = open_MRI_data_var(csv_path, train_set=0.8, normalize=True)

    # List of (nt, nfeatures) numpy objects
    nfeatures = X_train[0].shape[1]
    print(nfeatures)

    # Apply padding to both X_train and X_val
    X_train_tensor = [ torch.FloatTensor(t) for t in X_train ]
    X_train_pad = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False, padding_value=np.nan)
    X_test_tensor = [ torch.FloatTensor(t) for t in X_test ]
    X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)

    # Those datasets are of size [Tmax, Batch_size, nfeatures]
    # Save mask to unpad later when testing
    mask_train = ~torch.isnan(X_train_pad)
    mask_test = ~torch.isnan(X_test_pad)

    #convert those NaN to zeros
    X_train_pad[torch.isnan(X_train_pad)] = 0
    X_test_pad[torch.isnan(X_test_pad)] = 0

    #Create the dataloaders
    # train_loader = torch.utils.data.DataLoader(dataset_train,
    #                batch_size=batch_size, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(dataset_test, 
    #               batch_size=batch_size, shuffle=True)

    # Define model and optimizer
    model = ModelRNNVAE(p["x_size"], p["h_size"], p["hidden"], p["n_layers"], 
                        p["hidden"], p["n_layers"], p["hidden"],
                        p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    model.optimizer = optimizer

    model = model.to(DEVICE)
    # Fit the model
    model.fit(X_train_tensor.to(DEVICE), X_test_tensor.to(DEVICE))

    ### After training, save the model!
    model.save(out_dir, 'model.pt')

    # Predict the reconstructions from X_val and X_train
    X_test_fwd = model.predict(X_test_pad.to(DEVICE))
    X_train_fwd = model.predict(X_train_pad.to(DEVICE))

    #Reformulate things
    X_train_fwd['xnext'] = np.array(X_train_fwd['xnext']).swapaxes(0,1)
    X_train_fwd['zx'] = np.array(X_train_fwd['zx']).swapaxes(0,1)
    X_test_fwd['xnext'] = np.array(X_test_fwd['xnext']).swapaxes(0,1)
    X_test_fwd['zx'] = np.array(X_test_fwd['zx']).swapaxes(0,1)

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


    z_train = X_train_fwd['zx']
    z_test = X_test_fwd['zx']
    z_train = [Z for (i, Z) in enumerate(z_train)]
    z_test = [Z for (i, Z) in enumerate(z_test)]    
    z = z_train + z_test

    # Dir for projections
    if not os.path.exists(out_dir + 'z_proj/'):
        os.makedirs(out_dir + 'z_proj/')

    #plot latent space
    for dim0 in range(p["z_dim"]):
        for dim1 in range(dim0, p["z_dim"]):
            if dim0 == dim1: continue   # very dirty
            plot_z_time_2d(z, len(X_train_pad), [dim0, dim1], out_dir + 'z_proj/', out_name=f'z_d{dim0}_d{dim1}')


    # TODO: THIS SAMPLING PROCEDURE NEEDS TO BE UPDATED
    nt = len(X_train_pad)
    nsamples = 1000
    X_sample = model.sample_latent(nsamples, nt)

    #Get the samples
    X_sample['x'] = np.array(X_sample['x']).swapaxes(0,1)
    X_sample['z'] = np.array(X_sample['z']).swapaxes(0,1)

    # Dir for projections
    if not os.path.exists(out_dir + 'z_proj_sampling/'):
        os.makedirs(out_dir + 'z_proj_sampling/')

    #plot latent space
    for dim0 in range(p["z_dim"]):
        for dim1 in range(dim0, p["z_dim"]):
            if dim0 == dim1: continue   # very dirty
            plot_z_time_2d(X_sample['z'], len(X_train_pad), [dim0, dim1], out_dir + 'z_proj_sampling/', out_name=f'z_d{dim0}_d{dim1}')



    loss = {
        "mse_train" : mse_train,
        "mse_test": mse_test,
        "loss_total": model.loss['total'][-1],
        "loss_kl": model.loss['kl'][-1],
        "loss_ll": model.loss['ll'][-1]
    }

    return loss


if __name__ == "__main__":
    ### Parameter definition
    params = {
        "x_size": 40,
        "h_size": 10,
        "z_dim": 5,
        "hidden": 5,
        "n_layers": 1,
        "n_epochs": 1500,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
    }

    out_dir = "experiments/MRI_padding_lin/"
    csv_path = "data/tadpole_mrionly.csv"
    loss = run_experiment(params, csv_path, out_dir)