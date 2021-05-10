"""
Script to evaluate things of synthetic data

Plot latent space, compare latent space, compare reconstruction...
"""


import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae_s
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from rnnvae.eval import eval_reconstruction, eval_prediction
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rnnvae.utils import pickle_load

def run_eval(out_dir, data_cols, dropout_threshold_test, output_to_file=False):
    """
    Main function to evaluate a model.

    Evaluate a trained model
    out_dir: directory where the model is and the results will be stored.
    data_cols: name of channels.
    dropout_threshold_test: threshold of the dropout
    """

    ch_bl = [] ##STORE THE CHANNELS THAT WE CONVERT TO LONG BUT WERE BL

    #Redirect output to the out dir
    if output_to_file:
        sys.stdout = open(out_dir + 'output.out', 'w')

    #load parameters
    p = eval(open(out_dir + "params.txt").read())

    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    #Load data
    Z_test = pickle_load(out_dir + f"ztest")
    X_test = pickle_load(out_dir + f"xtest")

    X_test_list = []
    mask_test_list = []

    # Process test set
    for x_ch in X_test[:p["n_channels"]]:
        X_test_tensor = [ torch.FloatTensor(t) for t in x_ch]
        X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
        mask_test = ~torch.isnan(X_test_pad)
        mask_test_list.append(mask_test.to(DEVICE))
        X_test_pad[torch.isnan(X_test_pad)] = 0
        X_test_list.append(X_test_pad.to(DEVICE))

    p["n_feats"] = [p["n_feats"] for _ in range(p["n_channels"])]

    X_test = X_test[:p["n_channels"]]

    # Prepare model
    # Define model and optimizer
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
        print(model.dropout_comp)
        model.dropout_threshold = dropout_threshold_test

    ####################################
    # IF DROPOUT, CHECK THE COMPONENTS AND THRESHOLD AND CHANGE IT
    ####################################

    ##TEST
    X_test_fwd = model.predict(X_test_list, mask_test_list, nt=p["ntp"])

    # Test the reconstruction and prediction

    ############################
    ## Test reconstruction for each channel, using the other one 
    ############################
    # For each channel
    results = np.zeros((len(X_test), len(X_test))) #store the results, will save later

    for i in range(len(X_test)):
        for j in range(len(X_test)):
            curr_name = p["ch_names"][i]
            to_recon = p["ch_names"][j]
            av_ch = [j]
            mae_rec = eval_reconstruction(model, X_test, X_test_list, mask_test_list, av_ch, i)
            results[i,j] = mae_rec
            # Get MAE result for that specific channel over all timepoints
            print(f"recon_{curr_name}_from{to_recon}_mae: {mae_rec}")

    df_crossrec = pd.DataFrame(data=results, index=p["ch_names"], columns=p["ch_names"])
    plt.tight_layout()
    ax = sns.heatmap(df_crossrec, annot=True, fmt=".2f", vmin=0, vmax=1)
    plt.savefig(out_dir + "figure_crossrecon.png")
    plt.close()
    # SAVE AS FIGURE
    df_crossrec.to_latex(out_dir+"table_crossrecon.tex")

    ############################
    ## Test reconstruction for each channel, using the rest
    ############################
    # For each channel
    results = np.zeros((len(X_test), 1)) #store the results, will save later

    for i in range(len(X_test)):
        av_ch = list(range(len(X_test))).remove(i)
        to_recon = p["ch_names"][i]
        mae_rec = eval_reconstruction(model, X_test, X_test_list, mask_test_list, av_ch, i)
        results[i] = mae_rec
        # Get MAE result for that specific channel over all timepoints
        print(f"recon_{to_recon}_fromall_mae: {mae_rec}")

    df_totalrec = pd.DataFrame(data=results.T, columns=p["ch_names"])

    # SAVE AS FIGURE
    df_totalrec.to_latex(out_dir+"table_totalrecon.tex")

    ###############################################################
    # PLOTTING, FIRST GENERAL PLOTTING AND THEN SPECIFIC PLOTTING #
    ###############################################################

    # Test the new function of latent space
    #NEED TO ADAPT THIS FUNCTION
    qzx_test = [np.array(x) for x in X_test_fwd['qzx']]

    ####IF DROPOUT, SELECT ONLY COMPS WITH DROPOUT > TAL
    if model.dropout:
        kept_comp = model.kept_components
    else:
        kept_comp = None

    # Now plot color by timepoint
    out_dir_sample = out_dir + 'zcomp_ch_tp/'
    if not os.path.exists(out_dir_sample):
        os.makedirs(out_dir_sample)

    pallete = sns.color_palette("viridis", p["ntp"])
    pallete_dict = {i:value for (i, value) in enumerate(pallete)}

    #plot_latent_space(model, qzx_test, p["ntp"], plt_tp='all',
    #                all_plots=True, uncertainty=False, comp=kept_comp, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)

    # TODO plot per subject
    # TODO plot correlation between Z_true and Z_hat, and show the results

## MAIN
## here we put the parameters when we directly run the script
if __name__ == "__main__":
    out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/metaexp_synth/_h_100_z_10/"
    data_cols = ["c1","c2"]
    dropout_threshold_test = 1.0

    run_eval(out_dir, data_cols, dropout_threshold_test)




