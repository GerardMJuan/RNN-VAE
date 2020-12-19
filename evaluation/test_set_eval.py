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
from rnnvae import rnnvae
from rnnvae.utils import load_multimodal_data
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

out_dir = "/homedtic/gmarti/CODE/RNN-VAE/experiments_mc_newloss/h_doublefix/"
test_csv = "/homedtic/gmarti/CODE/RNN-VAE/data/multimodal_no_petfluid_test.csv"
# data_cols = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
data_cols = ['_mri_vol']

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
model = rnnvae.MCRNNVAE(p["h_size"], p["hidden"], p["n_layers"], 
                        p["hidden"], p["n_layers"], p["hidden"],
                        p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], 
                        p["n_channels"], p["ch_type"], p["n_feats"], DEVICE, print_every=100, 
                        phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                        dropout=p["dropout"], dropout_threshold=p["drop_th"])
model = model.to(DEVICE)
model.load(out_dir+'model.pt')

####################################
# IF DROPOUT, CHECK THE COMPONENTS AND THRESHOLD AND CHANGE IT
####################################


##TEST
X_test_fwd = model.predict(X_test_list, nt=ntp)

# Test the reconstruction and prediction
######################
## Prediction of last time point
######################

X_test_list_minus = []
X_test_tensors = []
mask_test_list_minus = []
for x_ch in X_test:
    X_test_tensor = [ torch.FloatTensor(t[:-1,:]) for t in x_ch]
    X_test_tensor_full = [ torch.FloatTensor(t) for t in x_ch]
    X_test_tensors.append(X_test_tensor_full)
    X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
    mask_test = ~torch.isnan(X_test_pad)
    mask_test_list_minus.append(mask_test.to(DEVICE))
    X_test_pad[torch.isnan(X_test_pad)] = 0
    X_test_list_minus.append(X_test_pad.to(DEVICE))

# Run prediction
#this is terribly programmed holy shit
X_test_fwd_minus = model.predict(X_test_list_minus, nt=ntp)
X_test_xnext = X_test_fwd_minus["xnext"]


i = 0
# Test data without last timepoint
# X_test_tensors do have the last timepoint
for (X_ch, ch) in zip(X_test[:3], p["ch_names"][:3]):
    #Select a single channel
    print(f'testing for {ch}')
    y_true = [x[-1] for x in X_ch if len(x) > 1]
    last_tp = [len(x)-1 for x in X_ch] # last tp is max size of original data minus one
    y_pred = []
    # for each subject, select last tp
    j = 0
    for tp in last_tp:
        if tp < 1: continue # ignore tps with only baseline
        y_pred.append(X_test_xnext[i][tp, j, :])
        j += 1

    #Process it to predict it
    mae_tp_ch = mean_absolute_error(y_true, y_pred)
    #save the result
    print(f'pred_{ch}_mae: {mae_tp_ch}')
    i += 1

############################
## Test reconstruction for each channel, using the other one 
############################
# For each channel
if p["n_channels"] > 1:

    for i in range(len(X_test)):
        curr_name = p["ch_names"][i]
        av_ch = list(range(len(X_test)))
        av_ch.remove(i)
        # try to reconstruct it from the other ones
        ch_recon = model.predict(X_test_list, nt=ntp, av_ch=av_ch)
        #for all existing timepoints

        y_true = X_test[i]
        # swap dims to iterate over subjects
        y_pred = np.transpose(ch_recon["xnext"][i], (1,0,2))
        y_pred = [x_pred[:len(x_true)] for (x_pred, x_true) in zip(y_pred, y_true)]

        #prepare it timepoint wise
        y_pred = [tp for subj in y_pred for tp in subj]
        y_true = [tp for subj in y_true for tp in subj]

        mae_rec_ch = mean_absolute_error(y_true, y_pred)

        # Get MAE result for that specific channel over all timepoints
        print(f"recon_{curr_name}_mae: {mae_rec_ch}")


###############################################################
# PLOTTING, FIRST GENERAL PLOTTING AND THEN SPECIFIC PLOTTING #
###############################################################

# Test the new function of latent space
#NEED TO ADAPT THIS FUNCTION
qzx_test = [np.array(x) for x in X_test_fwd['qzx']]


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

plot_latent_space(model, qzx_test, ntp, classificator=classif_test, pallete_dict=pallete_dict, plt_tp='all',
                all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)

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
            all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)

out_dir_sample_t0 = out_dir + 'zcomp_ch_dx_t0/'
if not os.path.exists(out_dir_sample_t0):
    os.makedirs(out_dir_sample_t0)

plot_latent_space(model, qzx_test, ntp, classificator=DX_test, pallete_dict=pallete_dict, plt_tp=[0],
                all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample_t0 + '_test', mask=mask_test_list)

# Now plot color by timepoint
out_dir_sample = out_dir + 'zcomp_ch_tp/'
if not os.path.exists(out_dir_sample):
    os.makedirs(out_dir_sample)

classif_test = [[i for (i, x) in enumerate(elem)] for elem in Y_test["DX"]]

pallete = sns.color_palette("viridis", ntp)
pallete_dict = {i:value for (i, value) in enumerate(pallete)}

plot_latent_space(model, qzx_test, ntp, classificator=classif_test, pallete_dict=pallete_dict, plt_tp='all',
                all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)


# Plot specific latent spaces
