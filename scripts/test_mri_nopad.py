"""
Test the RNNVAE with real MRI longitudinal data.

However, we use sequences of fixed length and we do not introduce padding at all
"""

import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae
from rnnvae.utils import open_MRI_data
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_many_trajectories


def run_experiment(p, csv_path, out_dir):
    """
    Function to run the experiments.
    p contain all the hyperparameters needed to run the experiments
    We assume that all the parameters needed are present in p!!
    out_dir is the out directory
    #hyperparameters
    """

    # out_dir
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

    # LOAD DATA
    X_train, X_test, Y_train, Y_test = open_MRI_data(csv_path, train_set=0.9, normalize=True, return_covariates=True)
    
    #Combine test and train Y for later
    Y = {}
    for k in Y_train.keys():
        Y[k] = Y_train[k] + Y_test[k]
    
    # List of (nt, nfeatures) numpy objects
    p["x_size"] = X_train[0].shape[1]
    print(p["x_size"])

    # Apply padding to both X_train and X_val
    #Permute so that dimension is (tp, nbatch, feat)
    X_train_tensor = torch.FloatTensor(X_train).permute((1,0,2))
    X_test_tensor = torch.FloatTensor(X_test).permute((1,0,2))

    # Define model and optimizer
    model = rnnvae.ModelRNNVAE(p["x_size"], p["h_size"], p["hidden"], p["n_layers"], 
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
    X_test_fwd = model.predict(X_test_tensor.to(DEVICE))
    X_train_fwd = model.predict(X_train_tensor.to(DEVICE))

    #Reformulate things
    X_train_fwd['xnext'] = np.array(X_train_fwd['xnext']).swapaxes(0,1)
    X_train_fwd['z'] = np.array(X_train_fwd['z']).swapaxes(0,1)
    X_test_fwd['xnext'] = np.array(X_test_fwd['xnext']).swapaxes(0,1)
    X_test_fwd['z'] = np.array(X_test_fwd['z']).swapaxes(0,1)

    X_test_hat = X_test_fwd["xnext"]
    X_train_hat = X_train_fwd["xnext"]

    # Unpad using the masks
    #after masking, need to rehsape to (nt, nfeat)

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

    # Dir for projections
    sampling_path = 'z_proj_dx/'
    if not os.path.exists(out_dir + sampling_path):
        os.makedirs(out_dir + sampling_path)

    #plot latent space
    for dim0 in range(p["z_dim"]):
        for dim1 in range(dim0, p["z_dim"]):
            if dim0 == dim1: continue   # very dirty
            plot_z_time_2d(z, p["ntp"], [dim0, dim1], out_dir + sampling_path, c='DX', Y=Y, out_name=f'z_d{dim0}_d{dim1}')

    # Dir for projections
    sampling_path = 'z_proj_age/'
    if not os.path.exists(out_dir + sampling_path):
        os.makedirs(out_dir + sampling_path)

    #plot latent space
    for dim0 in range(p["z_dim"]):
        for dim1 in range(dim0, p["z_dim"]):
            if dim0 == dim1: continue   # very dirty
            plot_z_time_2d(z, p["ntp"], [dim0, dim1], out_dir + sampling_path, c='AGE', Y=Y, out_name=f'z_d{dim0}_d{dim1}')


    #Sampling
    # TODO: THIS NEEDS TO BE UPDATED
    nt = p['ntp']
    nsamples = 1000
    X_sample = model.sample_latent(nsamples, nt)

    #Get the samples
    X_sample['xnext'] = np.array(X_sample['xnext']).swapaxes(0,1)
    X_sample['z'] = np.array(X_sample['z']).swapaxes(0,1)

    # Dir for projections
    sampling_path = 'z_proj_sampling/'
    if not os.path.exists(out_dir + sampling_path):
        os.makedirs(out_dir + sampling_path)

    #plot latent space
    for dim0 in range(p["z_dim"]):
        for dim1 in range(dim0, p["z_dim"]):
            if dim0 == dim1: continue   # very dirty
            plot_z_time_2d(X_sample['z'], p["ntp"], [dim0, dim1], out_dir + 'z_proj_sampling/', out_name=f'z_d{dim0}_d{dim1}')


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
        "n_epochs": 100,
        "clip": 10,
        "learning_rate": 1e-3,
        "ntp": 5,
        "batch_size": 128,
        "seed": 1714,
    }

    out_dir = "experiments/MRI_nopadding_lin/"
    csv_path = "data/tadpole_mrionly.csv"
    loss = run_experiment(params, csv_path, out_dir)