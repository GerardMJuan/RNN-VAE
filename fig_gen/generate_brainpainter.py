"""
Manually generate channels from baseline and 
"""
import sys
sys.path.insert(0, '/homedtic/gmarti/CODE/RNN-VAE/')
from rnnvae.utils import load_multimodal_data
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from rnnvae import rnnvae_h, rnnvae_s
import torch

#IMPORT FROM THE TEST CONFIG WE WANT
from fig_gen.configs import testconfig1 as params


# function to prepare data for brainpainter
def prepare_brainpainter_data(in_csv_train, suffix, synth_data, out_dir, name_fig):
    """
    Thsi function needs to:

    1. prepare the data wtih the correct format for brainpainter
    2. prepare the correct scale (infer it from a training set?)
    3. Save it to disk
    """

    # Get scale of CN values

    df_train = pd.read_csv(in_csv_train)
    cols = df_train.columns.str.contains('|'.join(suffix))
    cols = df_train.columns[cols].values
    #Generate mean and std of those features

    #load the data
    norm_val = pickle.load( open(f"data/norm_values/brainpainter_norm.pkl", 'rb'))
    mean_cn = norm_val["mean"]
    std_cn = norm_val["std"]

    # we compute mean of ad and mean of cn to discover the sign of the trajectory
    # for each biomarker (which should always be negative but i dont know anymore)

    #apply mean and std to the synth_data
    synth_data = - (synth_data - mean_cn) / std_cn

    #data should be between 0 and 3. How to solve this?
    #can mask so that less than 0 is 0, and higher than 3 is 3
    synth_data[synth_data < 0] = 0
    synth_data[synth_data > 3] = 3

    # create dataframe with the original columns (we already have the
    # translation in the config.py of the brainpainter)
    # "Image-name-unique" has to be the name of the first column (index)
    df_brainpainter = pd.DataFrame(data=synth_data,
                                   index= [f"img{i}" for i in range(len(synth_data))], 
                                   columns=cols)
    df_brainpainter.index.name = "Image-name-unique"

    # save to disk
    df_brainpainter.to_csv(out_dir + f"{name_fig}.csv")

if __name__ == "__main__":
    #paths
    out_dir = params.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_csv_train = params.in_csv_train
    in_csv_test = params.in_csv_test
    # out_dir = "/homedtic/gmarti/CODE/RNN-VAE/fig_gen/"
    # in_csv_train = "data/multimodal_no_petfluid.csv"
    #in_csv_test = "data/subj_for_brainpainter.csv"

    # Load the data
    # load full dataset and load the training dataset
    channels = ['_mri_vol','_mri_cort', '_cog']#, '_demog', '_apoe']
    names = ["MRI vol", "MRI cort", "Cog",]# "Demog", 'APOE']
    ch_type = ["long", "long", "long"]#, "bl", 'bl']
    X_train, _, Y_train, _, cols = load_multimodal_data(in_csv_train, channels, ch_type, train_set=1.0, normalize=True, return_covariates=True)
    ## Load the test data
    X_test, _, Y_test, _, cols = load_multimodal_data(in_csv_test, channels, ch_type, train_set=1.0, normalize=True, return_covariates=True)

    # load the model
    model_dir = params.model_dir 

    p = eval(open(model_dir + "params.txt").read())
    print(p)
    p["n_feats"] = [x[0].shape[1] for x in X_train]

    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    # either "old" or "new"
    type_model = params.type_model
    if type_model == "old":
        #  OLD VERSION
        model = rnnvae_h.MCRNNVAE(p["h_size"], p["hidden"], p["n_layers"], 
                                p["hidden"], p["n_layers"], p["hidden"],
                                p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                                p["clip"], p["n_epochs"], p["batch_size"], 
                                p["n_channels"], p["ch_type"], p["n_feats"], [None, None, None, None, None], DEVICE, print_every=100, 
                                phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                                dropout=p["dropout"], dropout_threshold=p["drop_th"])
    else:
        model = rnnvae_s.MCRNNVAE(p["h_size"], p["enc_hidden"],
                                p["enc_n_layers"], p["z_dim"], p["dec_hidden"], p["dec_n_layers"],
                                p["clip"], p["n_epochs"], p["batch_size"], 
                                p["n_channels"], p["ch_type"], p["n_feats"], p["c_z"], DEVICE, print_every=100, 
                                phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                                dropout=p["dropout"], dropout_threshold=p["drop_th"])

    model = model.to(DEVICE)
    model.load(model_dir+'model.pt')

    # prepare the two synthetic datas
    for i in range(len(X_test[0])):

        X_test_subj = [x[i] for x in X_test]
        X_test_list = []
        mask_test_list = []

        # Process test set
        # ONLY ONE SUBJECT!
        for x_ch in X_test_subj:
            X_test_tensor = [ torch.FloatTensor(x_ch)]
            X_test_pad = torch.nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
            mask_test = ~torch.isnan(X_test_pad)
            mask_test_list.append(mask_test.to(DEVICE))
            X_test_pad[torch.isnan(X_test_pad)] = 0
            X_test_list.append(X_test_pad.to(DEVICE))

        ntp = X_test_subj[0].shape[0]

        av_ch = params.av_ch
        X_test_fwd = model.predict(X_test_list, mask_test_list, ntp, av_ch, task='recon')
        X_test_fwd = X_test_fwd["xnext"]
        #Remove normalization
        norm_vol = pickle.load( open(f"data/norm_values/_mri_vol_norm.pkl", 'rb'))
        norm_cort = pickle.load( open(f"data/norm_values/_mri_cort_norm.pkl", 'rb'))

        X_subj_vol = (X_test_fwd[0].squeeze() * norm_vol["std"]) + norm_vol["mean"]
        X_subj_cort = (X_test_fwd[1].squeeze() * norm_cort["std"]) + norm_cort["mean"]
        subj_to_plot = np.concatenate((X_subj_vol, X_subj_cort),axis=1)

        ##############
        ###### TESTING
        ##############
        col_type = params.col_type
        # if ['_mri_vol','_mri_cort'] in col_type:
            # save it to disk using the brainpainter function
        prepare_brainpainter_data(in_csv_train, col_type, subj_to_plot, out_dir, f'subj{i}')
        # Save the rest of values normally