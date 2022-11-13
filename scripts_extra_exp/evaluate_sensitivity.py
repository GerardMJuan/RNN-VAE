"""
Evaluate sensitivity by computing the jacobian.
Using the jacobian, we can compute the sensitivity of the output to the input.
The sensitivity is computed as the sum of the absolute values of the jacobian.
The sensitivity is computed for each channel and each feature.
The sensitivity is computed for each sample, and it is averaged over the samples per diagnosis label.

This script is separate, as to not contaminate the main script with the revision code.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from rnnvae import rnnvae_s
import configparser
import time
from datetime import timedelta
import pandas as pd
from rnnvae.utils import load_multimodal_data

# Paths and parameters 
model_dir = '/homedtic/gmarti/EXPERIMENTS_MCVAE/final_hyperparameter_search/_h_50_z_30_hid_50_n_0/'
model_path = f'{model_dir}model.pt'
out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/revision_results/"

seed = 1714

# Data paths
train_path = "data/multimodal_no_petfluid_train.csv"
test_csv = "data/multimodal_no_petfluid_test.csv"

# create output dir if doesnt exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#Set CUDA
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)

# SET PARAMETERS OF THE ACTUAL MODEL
channels = ['_mri_vol','_mri_cort', '_cog']#, '_demog', '_apoe']
names = ["MRI vol", "MRI cort", "Cog"]#, "Demog", 'APOE']
ch_type = ["long", "long", "long"]#, "bl", 'bl']
constrain1=[None, None, 5]#, 5, 5]

p = {
    "h_size": 50,
    "z_dim": 30,
    "x_hidden": 10,
    "x_n_layers": 1,
    "z_hidden": 15,
    "z_n_layers": 1,
    "enc_hidden": 50,
    "enc_n_layers": 0,
    "dec_hidden": 10,
    "dec_n_layers": 0,
    "n_epochs": 3500,
    "clip": 10,
    "learning_rate": 2e-3,
    "batch_size": 128,
    "seed": 1714,
    "c_z": constrain1,
    "n_channels": len(channels),
    "ch_names" : names,
    "ch_type": ch_type,
    "phi_layers": True,
    "sig_mean": False,
    "dropout": True,
    "drop_th": 0.2,
    "long_to_bl": True
}

#Seed
torch.manual_seed(p["seed"])
np.random.seed(p["seed"])

# Now, load the data
#####################
### TRAIN DATA ######
#####################
X_train, _, Y_train, _, mri_col = load_multimodal_data(train_path, channels, p["ch_type"], train_set=1.0, normalize=True, return_covariates=True)

ntp_tr = max(np.max([[len(xi) for xi in x] for x in X_train]), np.max([[len(xi) for xi in x] for x in X_train]))

p["n_feats"] = [x[0].shape[1] for x in X_train]

X_train_list = []
mask_train_list = []


if p["long_to_bl"]:
    # HERE, change bl to long and repeat the values at t0 for ntp
    for i in range(len(p["ch_type"])):
        if p["ch_type"][i] == 'bl':
            for j in range(len(X_train[i])):
                X_train[i][j] = np.array([X_train[i][j][0]]*ntp) 

            # p["ch_type"][i] = 'long'

#For each channel, pad, create the mask, and append
for x_ch in X_train:
    X_train_tensor = [ torch.FloatTensor(t) for t in x_ch]
    X_train_pad = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False, padding_value=np.nan)
    mask_train = ~torch.isnan(X_train_pad)
    mask_train_list.append(mask_train.to(DEVICE))
    X_train_pad[torch.isnan(X_train_pad)] = 0
    X_train_pad.requires_grad = True
    X_train_pad.retain_grad()
    X_train_list.append(X_train_pad.to(DEVICE))

#####################
### TEST DATA ######
#####################
ch_bl_test = [] ##STORE THE CHANNELS THAT WE CONVERT TO LONG BUT WERE BL

X_test, _, Y_test, _, col_lists = load_multimodal_data(test_csv, channels, p["ch_type"], train_set=1.0, normalize=True, return_covariates=True)
import pdb; pdb.set_trace()

# need to deal with ntp here
ntp_test = max(np.max([[len(xi) for xi in x] for x in X_test]), np.max([[len(xi) for xi in x] for x in X_test]))

ntp = max(ntp_tr, ntp_test)

if p["long_to_bl"]:
    # Process MASK WITHOUT THE REPETITION OF BASELINE
    # HERE, change bl to long and repeat the values at t0 for ntp
    for i in range(len(p["ch_type"])):
        if p["ch_type"][i] == 'bl':

            for j in range(len(X_test[i])):
                X_test[i][j] = np.array([X_test[i][j][0]]*ntp) 

            # p["ch_type"][i] = 'long'
            ch_bl_test.append(i)

X_test_list = []
mask_test_list = []

# Process test set
for x_ch in X_test:
    X_test_tensor = [ torch.FloatTensor(t) for t in x_ch]
    X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
    mask_test = ~torch.isnan(X_test_pad)
    mask_test_list.append(mask_test.to(DEVICE))
    X_test_pad[torch.isnan(X_test_pad)] = 0
    X_test_pad.requires_grad = True
    X_test_pad.retain_grad()
    X_test_list.append(X_test_pad.to(DEVICE))

print('X_train')
#cada llista es un canal
# i es de size ntp, Nsubj, feats

print(len(X_train_list))
print(X_train_list[0].shape)

#Y train es un diccionary de 7 elements, i l'important és 'DX'

print('X_test')
print(len(X_test_list))
print(X_test_list[0].shape)

X_total = []
for Xtrain, Xtest in zip(X_train_list, X_test_list):
    X_total.append(torch.cat((Xtrain, Xtest), dim=1))

mask_total = []
for Mtrain, Mtest in zip(mask_train_list, mask_test_list):
    mask_total.append(torch.cat((Mtrain, Mtest), dim=1))

# Now, load the model (the final version)
model = rnnvae_s.MCRNNVAE(p["h_size"], p["enc_hidden"],
                        p["enc_n_layers"], p["z_dim"], p["enc_hidden"], p["enc_n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], 
                        p["n_channels"], p["ch_type"], p["n_feats"], p["c_z"], DEVICE, print_every=100, 
                        phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                        dropout=p["dropout"], dropout_threshold=p["drop_th"])

model = model.to(DEVICE)
model.load(model_path)

# Here, check if everything was done correctly

###JACOBIAN LOOP
jacobian = model.jacobian(X_total, mask_total, nt=ntp)
# El shape de cada jacobian es igual que l'imput

# Average over the subjects dividing first by timepoints and then by subjects,
#First applying the mask
mask_total = [x.cpu().numpy() for x in mask_total]

#Now, create three averaged jacobians, one for each diagnosis (AD, MCI, CN), which are in
# Y_train and Y_test, key "DX"
# First, get the y_train and y_test
y_train = Y_train['DX']
y_test = Y_test['DX']

#append the labels
y_total = np.concatenate((y_train, y_test))

# to simplify, get only the first diagnosis
y_total = [y[0] for y in y_total]

#The list of the jacobians per channel
jacobian_ch0_list = [jacobian[0][0], jacobian[1][0], jacobian[2][0]]
jacobian_ch1_list = [jacobian[0][1], jacobian[1][1], jacobian[2][1]]
jacobian_ch2_list = [jacobian[0][2], jacobian[1][2], jacobian[2][2]]

jacobian_ch0_output = []
jacobian_ch1_output = []
jacobian_ch2_output = []

for jacobian_list, output in zip([jacobian_ch0_list, jacobian_ch1_list, jacobian_ch2_list], [jacobian_ch0_output,jacobian_ch1_output,jacobian_ch2_output]):
    jacobian_cols = [] #put it here, very inneficient but fuck it
    jacobian_list = [x**2 for x in jacobian_list]

    # manually do the three 
    jacobian_masked_ch0 = jacobian_list[0].sum(axis=0)/mask_total[0].sum(axis=0)
    jacobian_masked_ch1 = jacobian_list[1].sum(axis=0)/mask_total[1].sum(axis=0)
    jacobian_masked_ch2 = jacobian_list[2].sum(axis=0)/mask_total[2].sum(axis=0)

    #Now, average over the subjects
    jacobian_ch0 = jacobian_masked_ch0.sum(axis=0)/jacobian_masked_ch0.shape[0]
    jacobian_ch1 = jacobian_masked_ch1.sum(axis=0)/jacobian_masked_ch1.shape[0]
    jacobian_ch2 = jacobian_masked_ch2.sum(axis=0)/jacobian_masked_ch2.shape[0]

    #save this results in a list
    output.append(jacobian_ch0)
    output.append(jacobian_ch1)
    output.append(jacobian_ch2)

# TO plot, just plot each one individually in a subplot, and
# use col_lists to get the names of the columns
# we also need to sort them beforehand
import matplotlib.pyplot as plt
import seaborn as sns

#use seaborn style for paper
sns.set_style("whitegrid")
sns.set_context("paper")

fig, axs = plt.subplots(3, 3, figsize=(15,10), sharex=False, sharey=False)
fig.suptitle('Sensitivity analysis: Cross-channel jacobian', fontsize=16)

col_lists[0] = [x.rstrip('_mri_vol') for x in col_lists[0]]
col_lists[1] = [x.rstrip('_mri_cort') for x in col_lists[1]]
col_lists[2] = [x.rstrip('_cog') for x in col_lists[2]]

#### AXES ARE TRANPOSED TO PLOT SIMILAR TO THE TABLES IN THE PAPER
axs = axs.T

## 0,0: Channel 0 from channel 0
ch00, col_lists_ch = zip(*sorted(zip(jacobian_ch0_output[0], col_lists[0].copy()), reverse=True))
sns.barplot(x=np.array(ch00[:15]), y=np.array(col_lists_ch[:15]), ax=axs[0][0], color=sns.color_palette("Set2")[0])
axs[0][0].set_title('Subcortical volumes')
# set name of y label to name the whole row
axs[0][0].set_ylabel('Vol.')
# axs[0][0].set_yticklabels(axs[0][0].get_yticklabels(), rotation=30)

## 0,1: Channel 0 from channel 1
ch01, col_lists_ch = zip(*sorted(zip(jacobian_ch0_output[1], col_lists[1].copy()), reverse=True))
sns.barplot(x=np.array(ch01[:15]), y=np.array(col_lists_ch[:15]), ax=axs[0][1], color=sns.color_palette("Set2")[0])
axs[1][0].set_title('Cortical thickness')
# axs[0][1].set_yticklabels(axs[0][1].get_yticklabels(), rotation=30)

## 0,2: Channel 0 from channel 2
ch02, col_lists_ch = zip(*sorted(zip(jacobian_ch0_output[2], col_lists[2].copy()), reverse=True))
sns.barplot(x=np.array(ch02[:15]), y=np.array(col_lists_ch[:15]), ax=axs[0][2], color=sns.color_palette("Set2")[0])
axs[2][0].set_title('Cognitive scores')
# axs[0][2].set_yticklabels(axs[0][2].get_yticklabels(), rotation=30)

## 1,0: Channel 1 from channel 0
ch10, col_lists_ch = zip(*sorted(zip(jacobian_ch1_output[0], col_lists[0].copy()), reverse=True))
sns.barplot(x=np.array(ch10[:15]), y=np.array(col_lists_ch[:15]), ax=axs[1][0], color=sns.color_palette("Set2")[1])
# axs[0][0].set_title('ch0 from ch0')
axs[0][1].set_ylabel('Cort.')
# set yticklabels to 60º
# axs[1][0].set_yticklabels(axs[1][0].get_yticklabels(), rotation=30)

## 1,1: Channel 1 from channel 1
ch11, col_lists_ch = zip(*sorted(zip(jacobian_ch1_output[1], col_lists[1].copy()), reverse=True))
sns.barplot(x=np.array(ch11[:15]), y=np.array(col_lists_ch[:15]), ax=axs[1][1], color=sns.color_palette("Set2")[1])
# axs[0][0].set_title('ch0 from ch0')
# set yticklabels to 60º
# axs[1][1].set_yticklabels(axs[1][1].get_yticklabels(), rotation=30)

## 1,2: Channel 1 from channel 2
ch12, col_lists_ch = zip(*sorted(zip(jacobian_ch1_output[2], col_lists[2].copy()), reverse=True))
sns.barplot(x=np.array(ch12[:15]), y=np.array(col_lists_ch[:15]), ax=axs[1][2], color=sns.color_palette("Set2")[1])
# axs[0][0].set_title('ch0 from ch0')
# set yticklabels to 60º
# axs[1][2].set_yticklabels(axs[1][2].get_yticklabels(), rotation=30)

## 2,0: Channel 2 from channel 0
ch20, col_lists_ch = zip(*sorted(zip(jacobian_ch2_output[0], col_lists[0].copy()), reverse=True))
sns.barplot(x=np.array(ch20[:15]), y=np.array(col_lists_ch[:15]), ax=axs[2][0], color=sns.color_palette("Set2")[2])
# axs[0][0].set_title('ch0 from ch0')
axs[0][2].set_ylabel('Cog.')
# set yticklabels to 60º
# axs[2][0].set_yticklabels(axs[2][0].get_yticklabels(), rotation=30)

## 2,1: Channel 2 from channel 0
ch21, col_lists_ch = zip(*sorted(zip(jacobian_ch2_output[1], col_lists[1].copy()), reverse=True))
sns.barplot(x=np.array(ch21[:15]), y=np.array(col_lists_ch[:15]), ax=axs[2][1], color=sns.color_palette("Set2")[2])
# axs[0][0].set_title('ch0 from ch0')
# set yticklabels to 60º
# axs[2][1].set_yticklabels(axs[2][1].get_yticklabels(), rotation=30)

## 2,2: Channel 2 from channel 0
ch22, col_lists_ch = zip(*sorted(zip(jacobian_ch2_output[2], col_lists[2].copy()), reverse=True))
sns.barplot(x=np.array(ch22[:15]), y=np.array(col_lists_ch[:15]), ax=axs[2][2], color=sns.color_palette("Set2")[2])
# axs[0][0].set_title('ch0 from ch0')
# set yticklabels to 60º
# axs[2][2].set_yticklabels(axs[2][2].get_yticklabels(), rotation=30)

plt.tight_layout()
plt.savefig(f'{out_dir}jacobian_ch.png', bbox_inches='tight')
plt.savefig(f'{out_dir}jacobian_ch.pdf', dpi=300, bbox_inches='tight')
plt.close()

#############################################################################################
#############################################################################################
#############################################################################################

##SEPARATE BY AD, CN,MCI
jacobian_ad_list = []
jacobian_mci_list = []
jacobian_cn_list = []
jacobian_cols = []
jacobian_total_list = []

# For each channel
for i in range(len(jacobian)):
    # COMPUTE SQUARE OF THE JACOBIAN
    jacobian[i] = [x**2 for x in jacobian[i]]

    jacobian_masked = [x*mask_total[i] for x in jacobian[i]]

    jacobian_masked = np.mean(jacobian_masked, axis=0)

    jacobian_masked = jacobian_masked.sum(axis=0)/mask_total[i].sum(axis=0)
    print(jacobian_masked.shape)

    #Then, average over timepoints
    # jacobian_masked = jacobian_masked.sum(axis=0)/jacobian_masked.shape[0]

    #Now, create the three jacobians
    jacobian_AD = jacobian_masked[np.array(y_total) == "AD"]
    jacobian_MCI = jacobian_masked[np.array(y_total) == "MCI"]
    jacobian_CN = jacobian_masked[np.array(y_total) == "CN"]

    #Now, average over the subjects
    jacobian_AD = jacobian_AD.sum(axis=0)/jacobian_AD.shape[0]
    jacobian_MCI = jacobian_MCI.sum(axis=0)/jacobian_MCI.shape[0]
    jacobian_CN = jacobian_CN.sum(axis=0)/jacobian_CN.shape[0]

    #save this results in a list
    jacobian_ad_list.append(jacobian_AD)
    jacobian_mci_list.append(jacobian_MCI)
    jacobian_cn_list.append(jacobian_CN)
    jacobian_cols.append(col_lists[i])

    # Do it also all together
    jacobian_total = jacobian_masked.sum(axis=0)/jacobian_masked.shape[0]
    jacobian_total_list.append(jacobian_total)


# convert lists of lists into a single list
jacobian_ad_list = [item for sublist in jacobian_ad_list for item in sublist]
jacobian_mci_list = [item for sublist in jacobian_mci_list for item in sublist]
jacobian_cn_list = [item for sublist in jacobian_cn_list for item in sublist]
jacobian_cols = [item for sublist in jacobian_cols for item in sublist]
jacobian_total_list = [item for sublist in jacobian_total_list for item in sublist]

#Now, sort the values for each jacobian, together with the col_lists at each channel
# and plot the results in a triple seaplot barplot

jacobian_ad_list, col_lists_ad = zip(*sorted(zip(jacobian_ad_list, jacobian_cols.copy()), reverse=True))
jacobian_mci_list, col_lists_mci = zip(*sorted(zip(jacobian_mci_list, jacobian_cols.copy()), reverse=True))
jacobian_cn_list, col_lists_cn = zip(*sorted(zip(jacobian_cn_list, jacobian_cols.copy()), reverse=True))
jacobian_total_list, col_lists_total = zip(*sorted(zip(jacobian_total_list, jacobian_cols.copy()), reverse=True))

#save the results in a dataframe and to disk
df = pd.DataFrame({'AD': jacobian_ad_list, 'AD_cols': col_lists_ad, 'MCI': jacobian_mci_list, 'col_list_MCI': col_lists_mci, 'CN': jacobian_cn_list, 'feature': col_lists_cn})
df.to_csv(f'{out_dir}/jacobian.csv', index=False)

# Do the triple barplot of the jacobians and col_lists in a single plot, using
# subplot with three horizontal plots
fig, axs = plt.subplots(3, 1, figsize=(10,7), sharex=True, sharey=False)
fig.suptitle('Jacobian of the RNN-VAE for each channel and diagnosis')

sns.barplot(x=np.array(jacobian_ad_list[:15]), y=np.array(col_lists_ad[:15]), ax=axs[0], color=sns.color_palette("hls", 8)[0])
axs[0].set_title('AD')
#set x axis labels at 45 degrees
#axs[0].tick_params(rotation=45)

sns.barplot(x=np.array(jacobian_mci_list[:15]), y=np.array(col_lists_mci[:15]), ax=axs[1], color=sns.color_palette("hls", 8)[1])
axs[1].set_title('MCI')
#set x axis labels at 45 degrees
#axs[1].tick_params(rotation=45)

sns.barplot(x=np.array(jacobian_cn_list[:15]), y=np.array(col_lists_cn[:15]), ax=axs[2], color=sns.color_palette("hls", 8)[2])
axs[2].set_title('CN')
#set x axis labels at 45 degrees
#axs[2].tick_params(rotation=45)

plt.savefig(f'{out_dir}jacobian.png', bbox_inches='tight')
plt.savefig(f'{out_dir}jacobian.pdf', dpi=300, bbox_inches='tight')
plt.close()

#Do the same, but only with jacobian_total (only one plot)
fig, axs = plt.subplots(1, 1, figsize=(10,7), sharex=True, sharey=False)
fig.suptitle('Jacobian of the RNN-VAE for each channel and diagnosis')

sns.barplot(x=np.array(jacobian_total_list[:15]), y=np.array(col_lists_total[:15]), ax=axs, palette='viridis')
axs.set_title('All subjects')
#set x axis labels at 45 degrees
#axs[0].tick_params(rotation=45)

plt.savefig(f'{out_dir}jacobian_total.png', bbox_inches='tight')
plt.savefig(f'{out_dir}jacobian_total.pdf', dpi=300, bbox_inches='tight')

plt.close()


