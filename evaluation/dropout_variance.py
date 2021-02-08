"""
Small script to test how the variance relates to the variational dropout

We open a trained model directory and check the variance associated with each latent dimension,

Focus on variance at t=0, but plot all

And maybe save it as an image for direct comparing?
"""

import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae_h
from rnnvae.utils import load_multimodal_data
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#PATH OF MODEL TO TEST
out_dir = "/homedtic/gmarti/EXPERIMENTS/RNNVAE/testing_longbl/dropout/"
test_csv = "/homedtic/gmarti/CODE/RNN-VAE/data/multimodal_no_petfluid_test.csv"
data_cols = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']

#load parameters
p = eval(open(out_dir + "params.txt").read())

# DEVICE
## Decidint on device on device.
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)

# Load test set
X_test, _, Y_test, _, col_lists = load_multimodal_data(test_csv, data_cols, p["ch_type"], train_set=1.0, normalize=True, return_covariates=True)
p["n_feats"] = [x[0].shape[1] for x in X_test]

X_test_list = []
mask_test_list = []

ntp = max(np.max([[len(xi) for xi in x] for x in X_test]), np.max([[len(xi) for xi in x] for x in X_test]))

# HERE, change bl to long and repeat the values at t0 for ntp
for i in range(len(p["ch_type"])):
    if p["ch_type"][i] == 'bl':
        #for j in range(len(X_train[i])):
        #    X_train[i][j] = np.array([X_train[i][j][0]]*ntp) 

        for j in range(len(X_test[i])):
            X_test[i][j] = np.array([X_test[i][j][0]]*ntp) 

        p["ch_type"][i] = 'long'

# Process test set
for x_ch in X_test:
    X_test_tensor = [ torch.FloatTensor(t) for t in x_ch]
    X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
    mask_test = ~torch.isnan(X_test_pad)
    mask_test_list.append(mask_test.to(DEVICE))
    X_test_pad[torch.isnan(X_test_pad)] = 0
    X_test_list.append(X_test_pad.to(DEVICE))

ntp = max([x.shape[0] for x in X_test_list])

print(p)
model = rnnvae_h.MCRNNVAE(p["h_size"], p["x_hidden"], p["x_n_layers"], 
                        p["z_hidden"], p["z_n_layers"], p["enc_hidden"],
                        p["enc_n_layers"], p["z_dim"], p["dec_hidden"], p["dec_n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], 
                        p["n_channels"], p["ch_type"], p["n_feats"], DEVICE, print_every=100, 
                        phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                        dropout=p["dropout"], dropout_threshold=p["drop_th"])

model = model.to(DEVICE)
model.load(out_dir+'model.pt')

# forward pass of test
X_test_fwd = model.predict(X_test_list, mask_test_list, nt=ntp)
###SAVE TO CSV, VARIANCES OF ALL TIMEPOINTS plus dropout
#do it on train or on test?

# each column is first the dropout, and then the variance of the component at each timepoint
df_data = pd.DataFrame()
dropout = model.dropout_comp.cpu().numpy().squeeze()
df_data["dropout"] = dropout

# for each channel, we need to add each timepoint
ch = 0
for z_ch in  X_test_fwd['zp']:
    ch_name = p['ch_names'][ch]
    t = 0
    for z_t in z_ch:
        if t == len(mask_test_list[ch]):
            break
        #get the scale, compute the mean across the subjects
        mask_for_z = mask_test_list[ch][t][:,0].repeat((z_t.scale.shape[1], 1)).T
        scale_masked = torch.reshape(torch.masked_select(z_t.scale, mask_for_z), (-1, z_t.scale.shape[1])) # apply the timepoint mask
        scale = scale_masked.mean(0).cpu().numpy().squeeze()
        df_data[f'ch{ch_name}_t{t}'] = scale
        t += 1
    ch += 1
df_data.to_csv(out_dir + 'variances.csv')

### SAVE IMAGE OF VAR T=0 AND DROPOUT AGAINST EACH OTHER.
# probably should do a subfigure, and a heatmap for each (as they will have
# different scales)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

sns.heatmap(df_data[['dropout']], annot=True, cmap='viridis', ax=ax1)

#heatmap for scales, need to unite all channels with t0
cols = []
for ch_name in p['ch_names']:
    cols.append(f'ch{ch_name}_t0')

sns.heatmap(df_data[cols], annot=True, cmap='viridis', ax=ax2)

plt.savefig(out_dir + 'variance_fig.png')