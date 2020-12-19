"""
Script that generates the latent figures from a 
multi-channel approach. 
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
test_csv = "/homedtic/gmarti/RNN-VAE/data/multimodal_no_petfluid_test.csv"
# data_cols = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
data_cols = ['_mri_vol']
#load parameters
#load parameters
p = eval(open(out_dir + "params.txt").read())
print(p['curves'])


# DEVICE
## Decidint on device on device.
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)

# Load test set
X_test, Y_test, mri_col = load_multimodal_data(test_csv, data_cols, p["ch_type"], train_set=1.0, normalize=True, return_covariates=True)

p["n_feats"] = [x[0].shape[1] for x in X_test]


X_train_list = []
mask_train_list = []

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

model = rnnvae.MCRNNVAE(p["h_size"], p["hidden"], p["n_layers"], 
                        p["hidden"], p["n_layers"], p["hidden"],
                        p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], 
                        p["n_channels"], p["ch_type"], p["n_feats"], DEVICE, print_every=100, 
                        phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                        dropout=p["dropout"], dropout_threshold=p["drop_th"])
model = model.to(DEVICE)
model.load(out_dir+'model.pt')

##TEST
X_test_fwd = model.predict(X_test_list, nt=ntp)

###############################################################
# PLOTTING, FIRST GENERAL PLOTTING AND THEN SPECIFIC PLOTTING #
###############################################################

# Test the new function of latent space
#NEED TO ADAPT THIS FUNCTION
qzx_test = [np.array(x) for x in X_test_fwd['qzx']]
