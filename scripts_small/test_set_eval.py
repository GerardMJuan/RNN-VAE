"""
Script to evaluate a specific model performance of a test set
"""

import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae_s
from rnnvae.utils import load_multimodal_data
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from rnnvae.eval import eval_reconstruction, eval_prediction
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eval(out_dir, test_csv, data_cols, dropout_threshold_test, type='test', output_to_file=False):
    """
    Main function to evaluate a model.

    Evaluate a trained model
    out_dir: directory where the model is and the results will be stored.
    test_csv: where the csv with the test data is stored.
    data_cols: name of channels.
    dropout_threshold_test: threshold of the dropout
    use_synth: use synthetic data
    """

    ch_bl = [] ##STORE THE CHANNELS THAT WE CONVERT TO LONG BUT WERE BL

    #load parameters
    p = eval(open(out_dir + "params.txt").read())

    long_to_bl = p["long_to_bl"] #variable to decide if we have transformed the long to bl or not.

    #Seed
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])

    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    if type == 'val':
        _, X_test, _, Y_test, col_lists = load_multimodal_data(test_csv, data_cols, p["ch_type"], train_set=0.8, normalize=True, return_covariates=True)
    else:
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

    # HACK: USING enc_n_layers twice, to make sure that encoder and decoder use the same number of layers
    #and to reduce redundant computations (as we are only working with )
    #same with enc_hidden
    model = rnnvae_s.MCRNNVAE(p["h_size"], p["enc_hidden"],
                            p["enc_n_layers"], p["z_dim"], p["enc_hidden"], p["enc_n_layers"],
                            p["clip"], p["n_epochs"], p["batch_size"], 
                            p["n_channels"], p["ch_type"], p["n_feats"], p["c_z"], DEVICE, print_every=100, 
                            phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                            dropout=p["dropout"], dropout_threshold=p["drop_th"])

    model = model.to(DEVICE)
    model.load(out_dir+'model.pt')

    # NOTE: AFTER LOADING THE MODEL, IF WE ARE TESTING DIRECTLY,
    # CHANGE THE OUT_DIR ADDING AN EXTRA DIRECTORY
    if type=='test':
        out_dir = out_dir + 'test-results/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    #Redirect output to the out dir
    if output_to_file:
        sys.stdout = open(out_dir + 'output.out', 'w')

    if p["dropout"]:
        print(model.dropout_comp)
        model.dropout_threshold = dropout_threshold_test


    ####################################
    # IF DROPOUT, CHECK THE COMPONENTS AND THRESHOLD AND CHANGE IT
    ####################################

    ##TEST
    X_test_fwd = model.predict(X_test_list, mask_test_list, nt=ntp)

    # Test the reconstruction and prediction

    ######################
    ## Prediction of last time point
    ######################
    # Test data without last timepoint
    # X_test_tensors do have the last timepoint
    pred_ch = list(range(3))
    t_pred = 1
    res = eval_prediction(model, X_test, t_pred, pred_ch, DEVICE)

    for (i,ch) in enumerate([x for (i,x) in enumerate(p["ch_names"]) if i in pred_ch]):
        print(f'pred_{ch}_mae, 1tp: {res[i]}')

    t_pred = 2
    res = eval_prediction(model, X_test, t_pred, pred_ch, DEVICE)

    for (i,ch) in enumerate([x for (i,x) in enumerate(p["ch_names"]) if i in pred_ch]):
        print(f'pred_{ch}_mae, 2tp: {res[i]}')

    t_pred = 3
    res = eval_prediction(model, X_test, t_pred, pred_ch, DEVICE)

    for (i,ch) in enumerate([x for (i,x) in enumerate(p["ch_names"]) if i in pred_ch]):
        print(f'pred_{ch}_mae, 3tp: {res[i]}')

    ############################
    ## Test reconstruction for each channel, using the other one 
    ############################
    # For each channel
    results = np.empty((len(X_test), len(X_test)), dtype=object) #store the results, will save later

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
    """
    plt.tight_layout()
    ax = sns.heatmap(df_crossrec, annot=True, fmt=".2f", vmin=0, vmax=1)
    plt.savefig(out_dir + "figure_crossrecon.png")
    plt.close()
    """
    # SAVE AS FIGURE
    df_crossrec = df_crossrec.round(decimals=2)
    df_crossrec.to_latex(out_dir+"table_crossrecon.tex")

    ############################
    ## Test reconstruction for each channel, using the rest
    ############################
    # For each channel
    results = np.empty((len(X_test), 1), dtype=object) #store the results, will save later

    for i in range(len(X_test)):
        av_ch = list(range(len(X_test))).remove(i)
        to_recon = p["ch_names"][i]
        mae_rec = eval_reconstruction(model, X_test, X_test_list, mask_test_list, av_ch, i)
        results[i] = mae_rec
        # Get MAE result for that specific channel over all timepoints
        print(f"recon_{to_recon}_fromall_mae: {mae_rec}")

    df_totalrec = pd.DataFrame(data=results.T, columns=p["ch_names"])

    # SAVE AS FIGURE
    df_totalrec_out = df_totalrec.round(decimals=2)
    df_totalrec_out.to_latex(out_dir+"table_totalrecon.tex")

    ###############################################################
    # PLOTTING, FIRST GENERAL PLOTTING AND THEN SPECIFIC PLOTTING #
    ###############################################################

    # Test the new function of latent space
    #NEED TO ADAPT THIS FUNCTION
    qzx_test = [np.array(x) for x in X_test_fwd['qzx']]

    # IF WE DO THAT TRANSFORMATION
    if long_to_bl:
        for i in ch_bl:
            qzx_test[i] = np.array([qzx if j == 0 else None for j, qzx in enumerate(qzx_test[i])])

    # Now plot color by timepoint
    out_dir_sample = out_dir + 'zcomp_ch_age/'
    if not os.path.exists(out_dir_sample):
        os.makedirs(out_dir_sample)

    #Binarize the ages and 
    age_full = [x for elem in Y_test["AGE_demog"] for x in elem]
    bins, retstep = np.linspace(min(age_full), max(age_full), 8, retstep=True)
    age_digitized = [np.digitize(y, bins) for y in Y_test["AGE_demog"]]

    classif_test = [[bins[x-1] for (i, x) in enumerate(elem)] for elem in age_digitized]

    pallete = sns.color_palette("viridis", 8)
    pallete_dict = {bins[i]:value for (i, value) in enumerate(pallete)}

    """
    ####IF DROPOUT, SELECT ONLY COMPS WITH DROPOUT > TAL
    if model.dropout:
        kept_comp = model.kept_components
    else:
        kept_comp = None

    print(kept_comp)
    plot_latent_space(model, qzx_test, ntp, classificator=classif_test, pallete_dict=pallete_dict, plt_tp='all',
                    all_plots=True, uncertainty=False, comp=kept_comp, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)

    #Convert to standard
    #Add padding so that the mask also works here
    DX_test = [[x for x in elem] for elem in Y_test["DX"]]

    #Define colors
    pallete_dict = {
        "CN": "#2a9e1e",
        "MCI": "#bfbc1a",
        "AD": "#af1f1f"
    }
    # Get classificator labels, for n time points
    out_dir_sample = out_dir + 'zcomp_ch_dx/'
    if not os.path.exists(out_dir_sample):
        os.makedirs(out_dir_sample)

    plot_latent_space(model, qzx_test, ntp, classificator=DX_test, pallete_dict=pallete_dict, plt_tp='all',
                all_plots=True, uncertainty=False, comp=kept_comp, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)

    out_dir_sample_t0 = out_dir + 'zcomp_ch_dx_t0/'
    if not os.path.exists(out_dir_sample_t0):
        os.makedirs(out_dir_sample_t0)

    plot_latent_space(model, qzx_test, ntp, classificator=DX_test, pallete_dict=pallete_dict, plt_tp=[0],
                    all_plots=True, uncertainty=False, comp=kept_comp, savefig=True, out_dir=out_dir_sample_t0 + '_test', mask=mask_test_list)

    # Now plot color by timepoint
    out_dir_sample = out_dir + 'zcomp_ch_tp/'
    if not os.path.exists(out_dir_sample):
        os.makedirs(out_dir_sample)

    classif_test = [[i for (i, x) in enumerate(elem)] for elem in Y_test["DX"]]

    pallete = sns.color_palette("viridis", ntp)
    pallete_dict = {i:value for (i, value) in enumerate(pallete)}

    plot_latent_space(model, qzx_test, ntp, classificator=classif_test, pallete_dict=pallete_dict, plt_tp='all',
                    all_plots=True, uncertainty=False, comp=kept_comp, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)
    """

    return df_totalrec

## MAIN
## here we put the parameters when we directly run the script
if __name__ == "__main__":

    out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/final_hyperparameter_search/_h_50_z_30_hid_50_n_0/"
    data_cols = ['_mri_vol','_mri_cort', '_cog']
    dropout_threshold_test = 0.2

    #out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/supplementary_6/"
    #out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/no_phi_x/test_sameparams/"
    test_csv = "/homedtic/gmarti/CODE/RNN-VAE/data/multimodal_no_petfluid_test.csv"
    #data_cols = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
    #dropout_threshold_test = 0.2

    run_eval(out_dir, test_csv, data_cols, dropout_threshold_test, 'test', output_to_file=True)