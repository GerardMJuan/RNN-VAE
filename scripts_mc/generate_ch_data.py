"""
Generate channel data from others.

Can be single channel (generate directly from the bojective channel) or multichannel
(addd with others)

As a parameter, include the time points to produce (similar to predict) and save it to disk 
at a .csv with the correct information)
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae
from rnnvae.utils import load_multimodal_data
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns



def run_experiment(p, csv_path, out_dir, out_csv_name, data_cols=[]):
    """
    Function to run the experiments.
    p contain all the hyperparameters needed to run the experiments
    We assume that all the parameters needed are present in p!!
    out_dir is the out directory
    out_csv_name is the 
    #hyperparameters
    """    


if __name__ == "__main__":

    ### Parameter definition

    #channels = ['_mri_vol','_mri_cort','_demog','_apoe', '_cog', '_fluid','_fdg','_av45']
    #names = ["MRI vol", "MRI cort", "Demog", "APOE", "Cog", "Fluid", "FDG", "AV45"]

    channels = ['_mri_vol','_mri_cort', '_cog', '_fluid']
    ch_gen = '_fluid'
    names = ["MRI vol", "MRI cort", "Cog", "Fluid"]
    ch_type = ["long", "long", "long", "long"]

    params = {
        "h_size": 10,
        "z_dim": 10,
        "hidden": 20,
        "n_layers": 0,
        "n_epochs": 500,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
        "n_channels": len(channels),
        "ch_names" : names,
        "ch_type": ch_type,
        "phi_layers": False,
        "sig_mean": False,
        "dropout": False,
        "drop_th": 0.2,
        ## here params only for this script
        "ntp_extra": 1,
        "ch_gen": ch_gen
    }

    out_dir = "experiments_mc/MRI_ADNI_full_tests/"
    out_csv_name = f"{ch_gen}data_generated.csv"
    csv_path = "data/multimodal_no_petfluid.csv"
    loss = run_experiment(params, csv_path, out_dir, out_csv_name, channels)