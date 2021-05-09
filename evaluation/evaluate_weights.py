"""
Script file that extracts and evaluate weights of the network and specific relationships between channels

For models that have only linear relationships, extract the weights of trained models and plot them.
Relationships between latent values and decoder, and between encoder and latent values.

This script will also serve to plot the desired figure relationships (have to design them)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from rnnvae import rnnvae_s
from rnnvae.utils import load_multimodal_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_weights_matrix(x, feat_list, z_list, out_dir, out_name):
    """
    function to plot the weights with a barplot, with a plot for each feature, 
    for a single latent dim, and showing positive negative magnitudes.
    """
    # Set up the matplotlib figure
    #f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    x_size = 0.3*len(feat_list)
    y_size = 0.3*len(z_list)
    plt.figure(figsize=(x_size,y_size))
    cmap = sns.diverging_palette(230, len(x), as_cmap=True)
    g = sns.heatmap(x, cmap=cmap, center=0, annot=False, linewidths=.5)#, cbar_kws={"shrink": .5})
    g.set_xlabel("Features")
    g.set_ylabel("Latent dim.")

    g.set_xticks(np.arange(len(x[0]))+0.5) # <--- set the ticks first
    g.set_xticklabels(feat_list, rotation=45, ha='right')

    g.set_yticks(np.arange(len(x))+0.5) # <--- set the ticks first
    g.set_yticklabels(z_list, rotation=45)#, ha='left')

    plt.tight_layout()
    plt.savefig(out_dir + out_name)
    plt.close()

def plot_weights_bar(x, feat_list, z_list, out_dir, out_name):
    """
    function to plot the weights with a matrix, for all features and latent points, using only color
    and value to account for the relat ionship
    """
    #f, ax = plt.subplots(figsize=(11, 9))
    #this shape should be depending on the length of the list
    x_size = 0.5*len(feat_list)
    y_size = 5
    plt.figure(figsize=(x_size,y_size))
    # Generate a custom diverging colormap
    pal = sns.diverging_palette(230, 20, n=len(x), as_cmap=False)
    rank = x.argsort().argsort()

    pallete = np.array(pal[::-1])[rank]
    g = sns.barplot(x=list(range(len(x))), y=x, ci=None, palette=pallete, dodge=False)
    #g.legend_.remove()
    g.set_xticks(np.arange(len(x))+0.5) # <--- set the ticks first
    g.set_xticklabels(feat_list, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(out_dir + out_name)
    plt.close()


def visualize_weights(out_dir, data_cols, dropout_threshold_test, output_to_file=False):
    """
    Function that gets the weights of each channel encoder/decoder and its relationship
    between each one and the latent values, to see the relatiionships. 

    After that, we can observe the cross-channel relationships, doing something like a group analysis.

    weights.data.numpy()

    """
    ch_bl = [] ##STORE THE CHANNELS THAT WE CONVERT TO LONG BUT WERE BL

    #Redirect output to the out dir
    if output_to_file:
        sys.stdout = open(out_dir + 'output.out', 'w')

    #load parameters
    p = eval(open(out_dir + "params.txt").read())

    long_to_bl = p["long_to_bl"] #variable to decide if we have transformed the long to bl or not.

    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    # get names of each feature
    test_csv = "/homedtic/gmarti/CODE/RNN-VAE/data/multimodal_no_petfluid_test.csv"
    X_test, _, _, _, feat_list = load_multimodal_data(test_csv, data_cols, p["ch_type"], train_set=1.0, normalize=True, return_covariates=True)

    p["n_feats"] = [x[0].shape[1] for x in X_test]

    # load model
    model = rnnvae_s.MCRNNVAE(p["h_size"], p["enc_hidden"],
                            p["enc_n_layers"], p["z_dim"], p["dec_hidden"], p["dec_n_layers"],
                            p["clip"], p["n_epochs"], p["batch_size"], 
                            p["n_channels"], p["ch_type"], p["n_feats"], p["c_z"], DEVICE, print_every=100, 
                            phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                            dropout=p["dropout"], dropout_threshold=p["drop_th"])

    model = model.to(DEVICE)
    model.load(out_dir+'model.pt')
    #CHANGE DROPOUT
    if p["dropout"]:
        model.dropout_threshold = 0.2

    enc_weights = []
    dec_weights = []

    comp_list = []

    for i in range(len(p["n_feats"])):
        # z components that are included
        kept_comp = model.kept_components

        weights_enc = model.ch_enc[i].to_mu.weight.data.cpu().detach().numpy()
        weights_dec = model.ch_dec[i].to_mu.weight.data.cpu().detach().numpy()

        #if there are restrictions, select only
        if p["c_z"][i]:
            kept_comp = [x for x in model.kept_components if x < p["c_z"][i]]

        # in the encoder, its size is feat_size + h_size. Need to select only feat_size
        #also, need to select only weights selected by DROPOUT and, for the channels that have 
        # restrictions, select only those. And, of course, need to keep labels.
        weights_enc = weights_enc[kept_comp,:p["n_feats"][i]]
        # in the decoder, its size is latent_size + h_size. Need to select only feat_size
        #also, need to select only weights selected by DROPOUT and, for the channels that have 
        # restrictions, select only those. And, of course, need to keep labels.
        weights_dec = weights_dec[:,:p["z_dim"]]
        weights_dec = weights_dec[:,kept_comp]

        enc_weights.append(weights_enc)
        dec_weights.append(weights_dec)

        comp_list.append(kept_comp)

    # For each latent space, plot relationship of each feature with that latent point
    # Do a matrix, or do a barplot? showing 
    weights_dir = out_dir + "matrixweights_out_dir/"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    #Remove suffix from feat_list
    i = 0
    for (feat, suffix) in zip(feat_list, data_cols):
        feat_list[i] = [x.replace(suffix,'') for x in feat]
        i += 1

    ##weight matrix
    for i in range(len(data_cols)):
        #encoder
        #transpose enc matrix so that we have the same shape for both
        plot_weights_matrix(enc_weights[i], feat_list[i], comp_list[i], weights_dir, f'w_enc_{data_cols[i]}')
        #decoder
        plot_weights_matrix(dec_weights[i].T, feat_list[i], comp_list[i], weights_dir, f'w_dec_{data_cols[i]}')
        break

    weights_dir = out_dir + "barweights_out_dir/"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # plot barplot
    # note that here, maybe the z_dim doesnt correspond to each other
    for i in range(len(data_cols)):
        
        #for that data col, select the correct z
        for z in range(len(comp_list[i])):
            #encoder
            plot_weights_bar(enc_weights[i][z], feat_list[i], [z], weights_dir, f'w_enc_{data_cols[i]}_z{comp_list[i][z]}')
            #decoder
            plot_weights_bar(dec_weights[i].T[z], feat_list[i], [z], weights_dir, f'w_dec_{data_cols[i]}_z{comp_list[i][z]}')
            break
        break

    ##### 
    # WEIGHTS ACROSS CHANNELS
    # Plot relationship between features across channels
    # by multiplying Enc_i x Dec_j, for each latent dimension they share

    weights_dir = out_dir + "crosschannels/"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    #for each combination of channels
    for i in range(len(data_cols)):
        for j in range(i, len(data_cols)):
            enc_w = enc_weights[i].T
            dec_w = dec_weights[j].T

            # if dim have different kept comps
            if comp_list[i] != comp_list[j]:
                # and between two sets
                comp_list_both = list(set(comp_list[i]) & set(comp_list[j]))
                # get indices of the dimensions that we want to conserve
                idx_i = [comp_list[i].index(idx) for idx in comp_list_both]
                idx_j = [comp_list[j].index(idx) for idx in comp_list_both]

                # apply those to the weights
                enc_w = enc_w[:,idx_i]
                dec_w = dec_w[idx_j,:]

            # multiply weights
            W =  np.matmul(enc_w, dec_w)
            # plot weights matrix with the new weights
            # TODO: NEED TO ADAPT SIZE TO THE ACTUAL LENGTH OF THE MATRIX
            plot_weights_matrix(W.T, feat_list[i], feat_list[j], weights_dir, f'w_crossch_{data_cols[i]}{data_cols[j]}')
            break
        break


    # TODO: FOR EACH Z_DIM SEPARATELY?
    # JUST DO THE SAME, BUT OVER A SINGLE Z_DIM AND JUST DO A MATMUL OVER THE 1D VECTORS
    weights_dir = out_dir + "crosschannels_zdim/"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    #for each combination of channels
    for i in range(len(data_cols)):
        for j in range(i, len(data_cols)):
            enc_w = enc_weights[i].T
            dec_w = dec_weights[j].T

            comp_list_both = comp_list[i]
            # if dim have different kept comps
            if comp_list[i] != comp_list[j]:
                # and between two sets
                comp_list_both = list(set(comp_list[i]) & set(comp_list[j]))
                # get indices of the dimensions that we want to conserve
                idx_i = [comp_list[i].index(idx) for idx in comp_list_both]
                idx_j = [comp_list[j].index(idx) for idx in comp_list_both]

                # apply those to the weights
                enc_w = enc_w[:,idx_i]
                dec_w = dec_w[idx_j,:]

            # For each shared component
            for comp_i in range(len(comp_list_both)):
                # multiply weights
                W =  np.outer(enc_w[:, comp_i], dec_w[comp_i, :])
                # plot weights matrix with the new weights
                plot_weights_matrix(W.T, feat_list[i], feat_list[j], weights_dir, f'w_crossch_{data_cols[i]}{data_cols[j]}_z{comp_list_both[comp_i]}')
            break
        break





## MAIN
## here we put the parameters when we directly run the script
if __name__ == "__main__":
    out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/SMALLNetwork_10/"
    data_cols = ['_mri_vol','_mri_cort', '_cog']
    dropout_threshold_test = 0.2

    visualize_weights(out_dir, data_cols, dropout_threshold_test)
