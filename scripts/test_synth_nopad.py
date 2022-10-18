"""
test_mc_synth.py

Testing for a single channel for synthetic,
longitudinal data. This file will be used. We will generate
different length signals to input to our model and see how it behaves.

Also, two different settings:
- generate them at different length but starting from the same time point
- generate them at different length and also at different time points.
"""

#imports
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae
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
    gen_model = SinDataGenerator(p["curves"], p["ntp"], p["noise"])
    samples = gen_model.generate_n_samples(p["nsamples"])
    X_train = np.asarray([y for (_,y) in samples])
    X_train_tensor = torch.FloatTensor(X_train).permute((1,0,2))

    samples = gen_model.generate_n_samples(int(p["nsamples"]*0.8))
    X_test = np.asarray([y for (_,y) in samples])
    X_test_tensor = torch.FloatTensor(X_test).permute((1,0,2))
    #Prepare model
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
    feature = 0
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
            plot_z_time_2d(z, p["ntp"], [dim0, dim1], out_dir + proj_path, out_name=f'z_d{dim0}_d{dim1}')

    #Sampling
    # Create first samples with only one timepoint
    gen_model = SinDataGenerator(p["curves"], p["ntp"], p["noise"])
    samples = gen_model.generate_n_samples(500)
    X_samples = np.asarray([y[:1] for (_,y) in samples])
    X_samples = torch.FloatTensor(X_samples).permute((1,0,2))

    X_sample = model.sequence_predict(X_samples.to(DEVICE), p['ntp'])

    #Get the samples
    X_sample['xnext'] = np.array(X_sample['xnext']).swapaxes(0,1)
    X_sample['z'] = np.array(X_sample['z']).swapaxes(0,1)

    # plot the samples over time
    plot_many_trajectories(X_sample['xnext'], 'all', p["ntp"], out_dir, 'x_samples')

    # Dir for projections
    sampling_path = 'z_proj_sampling/'
    if not os.path.exists(out_dir + sampling_path):
        os.makedirs(out_dir + sampling_path)

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