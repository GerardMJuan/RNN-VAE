"""
Script for evaluation of prediction of synth values, to compare one to another.

Yes, naming has no sense. test synth, evaluate synth... I'm sure I will regret that
months to come. However, it is what it is.

yes
"""


import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae_s
from rnnvae.utils import load_multimodal_data, denormalize_timepoint
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from rnnvae.eval import eval_reconstruction, eval_prediction
from sklearn.metrics import mean_squared_error, mean_absolute_error
from evaluate_cog_recon import create_trajectory_comp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from rnnvae.utils import pickle_load, pickle_dump

def run_traj(out_dir, data_cols, dropout_threshold_test, output_to_file=False):
    """
    Main function
    """

    #Redirect output to the out dir
    if output_to_file:
        sys.stdout = open(out_dir + 'output.out', 'w')


    #Load data
    Z_test = pickle_load(out_dir + f"ztest")
    X_test = pickle_load(out_dir + f"xtest")

    #load parameters
    p = eval(open(out_dir + "params.txt").read())

    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    # need to deal with ntp here
    ntp = max(np.max([[len(xi) for xi in x] for x in X_test]), np.max([[len(xi) for xi in x] for x in X_test]))

    X_test_list = []
    mask_test_list = []

    for x_ch in X_test[:p["n_channels"]]:
        X_test_tensor = [ torch.FloatTensor(t) for t in x_ch]
        X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
        mask_test = ~torch.isnan(X_test_pad)
        mask_test_list.append(mask_test.to(DEVICE))
        X_test_pad[torch.isnan(X_test_pad)] = 0
        X_test_list.append(X_test_pad.to(DEVICE))

    p["n_feats"] = [p["n_feats"] for _ in range(p["n_channels"])]

    model = rnnvae_s.MCRNNVAE(p["h_size"], p["enc_hidden"],
                            p["enc_n_layers"], p["z_dim"], p["dec_hidden"], p["dec_n_layers"],
                            p["clip"], p["n_epochs"], p["batch_size"], 
                            p["n_channels"], p["ch_type"], p["n_feats"], p["c_z"], DEVICE, print_every=100, 
                            phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                            dropout=p["dropout"], dropout_threshold=p["drop_th"])

    model.ch_name = p["ch_names"]

    model = model.to(DEVICE)
    model.load(out_dir+'model.pt')


    if p["dropout"]:
        model.dropout_threshold = dropout_threshold_test

    ## TEST WITH ONE, TWO OR THREE MISSING VALUES
    t_pred = 1

    # Run prediction
    X_test_fwd_minus = model.predict(X_test_list, mask_test_list, nt=p["ntp"])
    X_test_xnext = X_test_fwd_minus["xnext"]


    #variable to check if we are doing a denorm of the values or not
    denorm = False

    biomarkers = [i for i in range(p["n_feats"][0])]

    # MRI_VOL
    create_trajectory_comp(X_test[0], X_test_xnext[0], biomarkers, 'c1', out_dir, n=5, denorm=denorm)

    # MRI_CORT
    create_trajectory_comp(X_test[1], X_test_xnext[1], biomarkers, 'c2', out_dir, n=5, denorm=denorm)

    #### PASSAR-HO A FUNCIÓ I FER TOTS ELS CANALS I BIOMARCADORS, NORMALITZATS I NO NORMALITZATS
    create_trajectory_comp(X_test[2], X_test_xnext[2], biomarkers, 'c3', out_dir, n=5, denorm=denorm)

## MAIN
## here we put the parameters when we directly run the script
if __name__ == "__main__":
    out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/metaexp_synth/_h_50_z_30/"
    data_cols = ['c1','c2','c3']
    dropout_threshold_test = 1.0

    run_traj(out_dir, data_cols, dropout_threshold_test)


