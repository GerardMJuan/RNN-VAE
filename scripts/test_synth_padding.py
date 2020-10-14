"""
test_mc_synth.py

Testing for a single channel for synthetic,
longitudinal data. This file will be used. We will generate
signals of same length.

Also, two different settings:
"""

#imports
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
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_many_trajectories
from rnnvae.data_gen import SinDataGenerator


def run_experiment(p, out_dir):
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

    #Redirect output to the out dir
    sys.stdout = open(out_dir + 'output.out', 'w')

    #save parameters to the out dir 
    with open(out_dir + "params.txt","w") as f:
        f.write(str(p))

    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)


    #generate the data
    #Create class object
    gen_model = SinDataGenerator(p["curves"], p["ntp"], p["noise"], variable_tp=True)
    samples_train = gen_model.generate_n_samples(p["nsamples"])
    samples_test = gen_model.generate_n_samples(int(p["nsamples"]*0.8))

    X_train = [y for (_,y) in samples_train]
    X_test = [y for (_,y) in samples_test]
    
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

    #Prepare model
    # Define model and optimizer
    model = ModelRNNVAE(p["x_size"], p["h_size"], p["hidden"], p["n_layers"], 
                        p["hidden"], p["n_layers"], p["hidden"],
                        p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    model.optimizer = optimizer

    model = model.to(DEVICE)
    # Fit the model
    model.fit(X_train_pad.to(DEVICE), X_test_pad.to(DEVICE))

    ### After training, save the model!
    model.save(out_dir, 'model.pt')

    # Predict the reconstructions from X_val and X_train
    X_train_fwd = model.predict(X_train_pad.to(DEVICE))
    X_test_fwd = model.predict(X_test_pad.to(DEVICE))

    #Reformulate things
    X_train_fwd['xnext'] = np.array(X_train_fwd['xnext']).swapaxes(0,1)
    X_train_fwd['z'] = np.array(X_train_fwd['z']).swapaxes(0,1)
    X_test_fwd['xnext'] = np.array(X_test_fwd['xnext']).swapaxes(0,1)
    X_test_fwd['z'] = np.array(X_test_fwd['z']).swapaxes(0,1)

    X_test_hat = X_test_fwd["xnext"]
    X_train_hat = X_train_fwd["xnext"]

    # Unpad using the masks
    #after masking, need to rehsape to (nt, nfeat)
    X_test_hat = [X[mask_test[:,i,:]].reshape((-1, p["x_size"])) for (i, X) in enumerate(X_test_hat)]
    X_train_hat = [X[mask_train[:,i,:]].reshape((-1, p["x_size"])) for (i, X) in enumerate(X_train_hat)]

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
    feature = 4
    # For train
    plot_trajectory(X_train, X_train_hat, subj, 'all', out_dir, f'traj_train_s_{subj}_f_all') # testing for a given subject
    plot_trajectory(X_train, X_train_hat, subj, feature, out_dir, f'traj_train_s_{subj}_f_{feature}') # testing for a given feature

    # For test
    plot_trajectory(X_test, X_test_hat, subj, 'all', out_dir, f'traj_test_s_{subj}_f_all') # testing for a given subject
    plot_trajectory(X_test, X_test_hat, subj, feature, out_dir, f'traj_test_s_{subj}_f_{feature}') # testing for a given feature
    
    
    z_train = X_train_fwd['z']
    z_test = X_test_fwd['z']
    z_train = [Z for (i, Z) in enumerate(z_train)]
    z_test = [Z for (i, Z) in enumerate(z_test)]    
    z = z_train + z_test

    # Dir for projections
    proj_path = 'z_proj/'
    if not os.path.exists(out_dir + proj_path):
        os.makedirs(out_dir + proj_path)

    
    #plot latent space
    for dim0 in range(p["z_dim"]):
        for dim1 in range(dim0, p["z_dim"]):
            if dim0 == dim1: continue   # very dirty
            plot_z_time_2d(z, p["ntp"], [dim0, dim1], out_dir + 'z_proj/', out_name=f'z_d{dim0}_d{dim1}')
    
    #Sampling
    # Create first samples with all timepoints
    gen_model = SinDataGenerator(p["curves"], 1, p["noise"])
    samples = gen_model.generate_n_samples(500)
    X_samples = np.asarray([y for (_,y) in samples])
    X_samples = torch.FloatTensor(X_samples).permute((1,0,2))

    X_sample = model.sequence_predict(X_samples.to(DEVICE), p['ntp'])

    #Get the samples
    X_sample['xnext'] = np.array(X_sample['xnext']).swapaxes(0,1)
    X_sample['z'] = np.array(X_sample['z']).swapaxes(0,1)


    # Dir for projections
    sampling_path = 'z_proj_sampling/'
    if not os.path.exists(out_dir + sampling_path):
        os.makedirs(out_dir + sampling_path)

    # plot the samples over time
    plot_many_trajectories(X_sample['xnext'], 'all', p["ntp"], out_dir, 'x_samples')
    
    #plot latent space
    for dim0 in range(p["z_dim"]):
        for dim1 in range(dim0, p["z_dim"]):
            if dim0 == dim1: continue   # very dirty
            plot_z_time_2d(X_sample['z'], p["ntp"], [dim0, dim1], out_dir + sampling_path, out_name=f'z_d{dim0}_d{dim1}')
    
    loss = {
        "mse_train" : mse_train,
        "mse_test": mse_test,
        "loss_total": model.loss['total'][-1],
        "loss_kl": model.loss['kl'][-1],
        "loss_ll": model.loss['ll'][-1]
    }

    return loss


if __name__ == "__main__":
    curves = [
        ("sigmoid", {"L": 1, "k": 1, "x0": 5}),    
        ("sin", {"A": 1, "f": 0.2}),
        ("cos", {"A": 1, "f": 0.2}),
        ("sigmoid", {"L": 1, "k": -15, "x0": 5})
        ]

    ### Parameter definition
    params = {
        "x_size": len(curves),
        "h_size": 20,
        "z_dim": 5,
        "hidden": 20,
        "n_layers": 1,
        "n_epochs": 2000,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
        "curves": curves,
        "ntp": 15,
        "noise": 0.2,
        "nsamples": 300
    }

    out_dir = "experiments/synth_nopadding/"
    loss = run_experiment(params, out_dir)