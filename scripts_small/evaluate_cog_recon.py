"""
Script for evaluation of prediction of cognitive values.

We directly look at the values predicted to see if they differ a lot from the
actual real values, how do they differ, and evaluate whether categorizing
them has any real impact on the performance of the model (meaning that the model
only works well on the space points present in the training set, and thus, it does not generalize
in all the latent space manifold)
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

def create_trajectory_comp(X_test, X_test_hat, biomarkers, label, out_dir, n=15, denorm=False):

    """
    Function that creates a figure for each subject and biomarker combination.
    It creates the number of figures specified by n, a figure for each subject.
    """
    if denorm:
        out_folder = f"{label}_comparison_denorm/"
    else:
        out_folder = f"{label}_comparison/"

    if not os.path.exists(out_dir + out_folder):
        os.makedirs(out_dir + out_folder)

    x = X_test
    # reshape to have the same type of shape
    x_hat = np.swapaxes(np.array(X_test_hat), 0, 1)

    for i in range(len(biomarkers)):
        # for each subject
        j = 0

        list_subj = random.sample(range(len(x)), n)
        for j in list_subj:
            subj = x[j]
            subj_hat = x_hat[j]
            # normalize values if needed
            if denorm: 
                subj = denormalize_timepoint(subj, f'_{label}')
                subj_hat = denormalize_timepoint(subj_hat, f'_{label}')
            fig= plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel("Time-point")
            ax.set_ylabel("Value")
            fig.suptitle(f"Subject {j}")
            ax.plot(list(range(len(subj[:,i]))), subj[:,i], linewidth=2.5,c="b", label='real')
            ax.plot(list(range(len(subj_hat[:,i]))), subj_hat[:,i], linewidth=2.5,c="r", label='predicted')
            ax.legend()
            plt.savefig(out_dir + out_folder + f'{biomarkers[i]}_subj_{j}.png')
            plt.close()

if __name__ == "__main__":

    out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/SMALLNetwork_10/"
    test_csv = "/homedtic/gmarti/CODE/RNN-VAE/data/multimodal_no_petfluid_test.csv"
    data_cols = ['_mri_vol','_mri_cort', '_cog']#, '_demog', '_apoe']
    dropout_threshold_test = 0.1

    ch_bl = [] ##STORE THE CHANNELS THAT WE CONVERT TO LONG BUT WERE BL

    #load parameters
    p = eval(open(out_dir + "params.txt").read())

    long_to_bl = p["long_to_bl"] #variable to decide if we have transformed the long to bl or not.

    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    # Load test set
    X_test, _, Y_test, _, col_lists = load_multimodal_data(test_csv, data_cols, p["ch_type"], train_set=1.0, normalize=True, return_covariates=True)
    p["n_feats"] = [x[0].shape[1] for x in X_test]

    # need to deal with ntp here
    ntp = max(np.max([[len(xi) for xi in x] for x in X_test]), np.max([[len(xi) for xi in x] for x in X_test]))

    if long_to_bl: 
        # Process MASK WITHOUT THE REPETITION OF BASELINE
        # HERE, change bl to long and repeat the values at t0 for ntp
        for i in range(len(p["ch_type"])):
            if p["ch_type"][i] == 'bl':

                for j in range(len(X_test[i])):
                    X_test[i][j] = np.array([X_test[i][j][0]]*ntp) 

                # p["ch_type"][i] = 'long'
                ch_bl.append(i)

    X_test_list = []
    mask_test_list = []

    # Process test set
    for x_ch in X_test:
        X_test_tensor = [ torch.FloatTensor(t) for t in x_ch]
        X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
        mask_test = ~torch.isnan(X_test_pad)
        mask_test_list.append(mask_test.to(DEVICE))
        X_test_pad[torch.isnan(X_test_pad)] = 0
        X_test_list.append(X_test_pad.to(DEVICE))


    model = rnnvae_s.MCRNNVAE(p["h_size"], p["enc_hidden"],
                            p["enc_n_layers"], p["z_dim"], p["dec_hidden"], p["dec_n_layers"],
                            p["clip"], p["n_epochs"], p["batch_size"], 
                            p["n_channels"], p["ch_type"], p["n_feats"], p["c_z"], DEVICE, print_every=100, 
                            phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                            dropout=p["dropout"], dropout_threshold=p["drop_th"])

    model = model.to(DEVICE)
    model.load(out_dir+'model.pt')

    if p["dropout"]:
        model.dropout_threshold = dropout_threshold_test


    ## TEST WITH ONE, TWO OR THREE MISSING VALUES
    t_pred = 1

    #process the data
    X_test_minus = []
    mask_test_minus = []
    for x_ch in X_test:
        #Want to select only last T-t_pred values
        X_test_tensor = [ torch.FloatTensor(t[:-t_pred,:]) for t in x_ch]
        X_test_tensor_full = [ torch.FloatTensor(t) for t in x_ch]
        X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
        mask_test = ~torch.isnan(X_test_pad)
        mask_test_minus.append(mask_test.to(DEVICE))
        X_test_pad[torch.isnan(X_test_pad)] = 0
        X_test_minus.append(X_test_pad.to(DEVICE))

    #Number of time points is the length of the input channel we want to predict
    ntp = max(np.max([[len(xi) for xi in x] for x in X_test]), np.max([[len(xi) for xi in x] for x in X_test]))
    # Run prediction
    X_test_fwd_minus = model.predict(X_test_minus, mask_test_minus, nt=ntp)
    X_test_xnext = X_test_fwd_minus["xnext"]

    ## COMPARE THE PREDICTED WITH THE REAL VALUES
    ## PLOT THE COMPARISON FOR ALL THE SUBJECTS AND ALL
    # THE BIOMARKERS OF COG, APOE, AND YED/SEX
    # output folder

    #variable to check if we are doing a denorm of the values or not
    denorm = False

    cog_biomarkers = ["CDRSB_cog", "ADAS11_cog", "ADAS13_cog", "MMSE_cog", "RAVLT_immediate_cog", "FAQ_cog"]
    demog_biomarkers = ["AGE_demog", "PTGENDER_demog", "PTEDUCAT_demog"]
    apoe = ["APOE4_apoe"]

    # MRI_VOL
    create_trajectory_comp(X_test[0], X_test_xnext[0], col_lists[0], 'mri_vol', out_dir, n=5, denorm=denorm)

    # MRI_CORT
    create_trajectory_comp(X_test[1], X_test_xnext[1], col_lists[1], 'mri_cort', out_dir, n=5, denorm=denorm)

    #### COG
    #### PASSAR-HO A FUNCIÓ I FER TOTS ELS CANALS I BIOMARCADORS, NORMALITZATS I NO NORMALITZATS
    create_trajectory_comp(X_test[2], X_test_xnext[2], cog_biomarkers, 'cog', out_dir, n=5, denorm=denorm)

    #### DEMOG
    # create_trajectory_comp(X_test[3], X_test_xnext[3], demog_biomarkers, 'demog', out_dir, n=5, denorm=False)
