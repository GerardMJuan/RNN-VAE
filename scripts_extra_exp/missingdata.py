"""
Main script to test for performance changes with more percentage
of missing data.

For each percentage of missing data, the script will run the
models 10 times and average the results.

The models used are GFA, KNN, and MC-RVAE.

The training is with the complete data, the test is with the missing data.

And it tries to reconstruct in the same way as the original test script.
"""
# Imports
import sys
sys.path.insert(0, '/homedtic/gmarti/CODE/RNN-VAE/')
from rnnvae.utils import load_multimodal_data
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from itertools import zip_longest
import copy

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri

import torch
from torch import nn
from rnnvae import rnnvae_s
import time
import pickle 

#Set CUDA
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)

def my_mean_absolute_error(y_true, y_pred, is_GFA=False):
    """
    Auxiliar function to compute the mean and std of the absolute error
    """

    # get only the length of y_pred that also appear in y_true
    if not is_GFA: 
        y_pred_s = [x_pred[:min(len(x_true), len(x_pred))] for (x_pred, x_true) in zip(y_pred, y_true)]
        y_true_s = [x_true[:min(len(x_true), len(x_pred))] for (x_pred, x_true) in zip(y_pred, y_true)]
        y_pred = [tp for subj in y_pred_s for tp in subj]
        y_true = [tp for subj in y_true_s for tp in subj]
    
    maes_list = [mean_absolute_error(y_t, y_p) for (y_t, y_p) in zip(y_true, y_pred)]

    mean_mae = np.mean(maes_list)
    std_mae = np.std(maes_list)
    return (mean_mae, std_mae)


def knn_recon(X_train, X_test, n_ch, av_ch=None, n_neighbors=5):
    """
    Predict missing modality (indicated by n_ch) using a KNN approach for each modality. 
    the most similar one in the KNN space and averagin them to form the output
    we consider all timepoints as separate points for the knn
    later, to create the output, we just do the mean of all the channels, across all timepoints, even if they
    are variable.

    Differences: here we have missing data "inputted as NA", so that needs to be taken into accound too.

    Basically use: https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
    """

    tic = time.time()
    if av_ch is None:
        av_ch = list(range(len(X_test)))
        av_ch.remove(n_ch)

    y_pred = []
    X_missing_mod = X_train[n_ch]
    # needs to be same length as training_knn

    fitting_data = []
    X_missingmod_knn = []
    #Fit knn for each modality
    for (i, X_ch) in enumerate(X_train):

        training_knn = []
        corresponding_mod_knn = []
        #get data
        # only select 
        #for each subject, only select timepoints that also appear in the original data
        for (nsubj, x) in enumerate(X_ch): # for each subject
            for (tp, sample) in enumerate(x): #for each sample
                if tp < len(X_missing_mod[nsubj]):
                    training_knn.append(sample)
                    corresponding_mod_knn.append(X_missing_mod[nsubj][tp])

        fitting_data.append(training_knn)
        X_missingmod_knn.append(corresponding_mod_knn)

    # for every subject
    for subj_idx in range(len(X_test[0])):
        
        predicted_ch = []
        # for every channel
        for (i, X_ch) in enumerate(X_test):
            if i not in av_ch: continue # dont compute it if it is not in the available channels
            # if i == n_ch: continue # dont compute it over the missing modality
            curr_channel_pred = []

            # could change for another number of neighbors
            # knn_input = KNNImputer(n_neighbors=n_neighbors) #is this enough? test it
            # knn_input.fit(fitting_data[i])

            # Input the test data
            knn = NearestNeighbors(n_neighbors=n_neighbors) #is this enough? test it
            knn.fit(fitting_data[i])

            knn_input = KNNImputer(n_neighbors=n_neighbors) #is this enough? test it
            knn_input.fit(fitting_data[i])


            subj = X_ch[subj_idx]
            #for every timepoint
            for sample in subj:
                #find most similar subject in our training set
                sample = knn_input.transform(sample.reshape(1, -1))
                idx_list = knn.kneighbors(sample, return_distance=False)[0]
                neighbor = []
                for idx in idx_list:
                    #get the corresponding values of that modality
                    pred_sample = X_missingmod_knn[i][idx]
                    neighbor.append(pred_sample)
                pred_sample = np.mean(np.array(neighbor), axis=0)
                curr_channel_pred.append(pred_sample)

            predicted_ch.append(curr_channel_pred)

        #Now, need to compute the mean across the samples
        if len(predicted_ch) > 1: 
            subj_pred = list(zip_longest(*predicted_ch))
            # Compute the mean across every list (removing the Nones first)
            subj_pred = [[x for x in l if x is not None] for l in subj_pred]
            subj = [np.mean(p, axis=0) for p in subj_pred]
        else:
            subj = predicted_ch[0]
        y_pred.append(subj)

    tac = time.time()
    print("Time to predict using KNN: ", tac-tic)
    return y_pred


def prepare_gfa_recon_model(X_og, X_test):
    """
    Quick fucntion to prepare a model for reconstruction with the missing data

    Return the reconstructed data already
    """
    # Prepare the data in the correct format
    Y_og = [[] for _ in range(len(X_og))]
    for i in range(len(X_og[0])):
        # select only the min len of each subject
        min_len = np.min([len(x[i]) for x in X_og])
        for j in range(len(X_og)):
            Y_og[j].append(X_og[j][i][:min_len])

    # prepare it arranging the timepoints
    Y_og = [np.array([tp for subj in Y_ch for tp in subj]) for Y_ch in Y_og]

    # Prepare the data in the correct format
    Y_test = [[] for _ in range(len(X_test))]
    for i in range(len(X_test[0])):
        # select only the min len of each subject
        min_len = np.min([len(x[i]) for x in X_test])
        for j in range(len(X_test)):
            Y_test[j].append(X_test[j][i][:min_len])

    # prepare it arranging the timepoints
    Y_test = [np.array([tp for subj in Y_ch for tp in subj]) for Y_ch in Y_test]

    # Reconstruct first the missing data
    opts = gfa_package.getDefaultOpts()
    opts = gfa_package.informativeNoisePrior(Y_test, opts)
    opts.rx2['iter.burnin'] = 200.0
    opts.rx2['iter.max'] = 500.0

    gfa_model_recon = gfa_package.gfa(Y_test, opts, K=50)
    Y_test_recon = gfa_package.reconstruction(gfa_model_recon)

    # Divide in the different channels 40 - 68 - 6
    Y_test_recon_new = [Y_test_recon[:, :40], Y_test_recon[:, 40:108], Y_test_recon[:, 108:]]

    return Y_og, Y_test_recon_new


def gfa_recon(X_test, model, n_ch, av_ch=None):
    """
    Predict missing modality (indicated by n_ch) using GFA. 
    the most similar one in the KNN space and averagin them to form the output
    we consider all timepoints as separate points for the knn
    later, to create the output, we just do the mean of all the channels, across all timepoints, even if they
    are variable.

    Differences: here we have missing data "inputted as NA", so that needs to be taken into account too.
    Theoretically, GFA directly uses missing data, but need to make sure how the format is.
    """
    if av_ch is None:
        av_ch = list(range(len(X_test)))
        av_ch.remove(n_ch)

    # Run the model
    opts = model.rx2('opts')

    # if av_ch is only one, we need to define the channels differently
    # if not, we can just use the av_ch
    if len(av_ch) == 1:
        opts.rx2['prediction'] = np.array([av_ch[0] != ii for ii in range(len(X_test))])
    else:
        opts.rx2['prediction'] = np.array([n_ch == ii for ii in range(len(X_test))])

    model.rx2['opts'] = opts
    pred = gfa_package.sequentialGfaPrediction(X_test, model)

    # Get the predictions
    y_pred = pred[n_ch]
    return y_pred


def mcrvae_recon(X_test, mask_test, model, n_ch, av_ch=None):
    """
    Predict missing modality (indicated by n_ch) using a MC-RVAE approach for each modality. 
    the most similar one in the KNN space and averagin them to form the output
    we consider all timepoints as separate points for the knn
    later, to create the output, we just do the mean of all the channels, across all timepoints, even if they
    are variable.

    Differences: here we have missing data "inputted as NA", so that needs to be taken into account too.

    Probably need to create a new function or modifiy the reconstruction function to fill the NA 
    values with the prediction either from the prior or from the posterior.
    """
    assert model.is_fitted, "Model is not fitted!"

    # Prepare the data in the correct format
    model.dropout_threshold = 0.2
    
    ntp = len(X_test[n_ch])

    # This function needs an extra parameter to tell it that it should infer the missing values
    ch_recon = model.predict(X_test, mask_test, nt=ntp, av_ch=av_ch, task='recon', na=True)

    y_pred = np.transpose(ch_recon["xnext"][n_ch], (1,0,2)) # transpose to obtain dim that are comparable to X_test
    y_pred = [x_pred[:len(x_true)] for (x_pred, x_true) in zip(y_pred, y_true)]

    return y_pred


def apply_missing_data(X, perc):
    """
    Function that applies the percentage of missing data to the data
    
    X: has the format of the data in here, list of channels, each channel is a list of timepoints, 
    each subject is numpy array of nsubj*nfeat
    
    Apply the missing data at random across the nfeats and ntimepoints, but only on the ntimepoints
    according to the mask. 
    The missing data is defined as np.nan
    
    It returns X_missing, with the same exact size and shape as X, but with the missing data
    """
    X_missing = []
    for ch in X:
        ch_missing = []
        for subj in ch:
            # randomly select the features to remove
            nfeat = int(subj.shape[0]*subj.shape[1]*perc)
            mask = np.zeros(subj.shape[0]*subj.shape[1], dtype=bool)
            
            # marking first n indexes as true
            mask[:nfeat] = True
            
            # shuffling the mask
            np.random.shuffle(mask)
            mask = mask.reshape(subj.shape[0], subj.shape[1])
            subj[mask] = np.nan

            ch_missing.append(subj)
        X_missing.append(ch_missing)
    return X_missing

def convert_X_to_tensor(X):
    """
    Auxiliary function to convert the data to tensor format
    """
    #For each channel, pad, create the mask, and append
    X_list = []
    mask_list = []

    for x_ch in X:
        X_tensor = [ torch.FloatTensor(t) for t in x_ch]
        X_pad = nn.utils.rnn.pad_sequence(X_tensor, batch_first=False, padding_value=np.nan)
        mask = ~torch.isnan(X_pad)
        mask_list.append(mask.to(DEVICE))
        X_pad[torch.isnan(X_pad)] = 0
        X_pad.requires_grad = True
        X_pad.retain_grad()
        X_list.append(X_pad.to(DEVICE))

    return X_list, mask_list

def mising_data_to_tensor(X, X_missing, mask):
    """
    X: list of channels, each channel being a list of timepoints, each timepoint being a pytorch Tensor (nsubj, nfeat). It has padding,
    so it is not the same as X. arrays with padding are all zero.
    X_missing: list of channels, each channel being a list of subjects, each subject being a numpy array (ntp, nfeat)
    mask: list of channels, each channel being a list of timepoints, each timepoint being a pytorch Tensor (nsubj, nfeat). It indicates which timepoints are missing.

    X_missing contains NA values at random across the timepoints and features, but only for the timepoints indicated by the mask.

    This function returns X_missing_tensor, which is the same as X, but with the corresponding NA defined by X_missing.
    """

    for x_ch, mask_ch, x_ch_missing in zip(X, mask, X_missing):
        for tp in range(x_ch.shape[0]):
            for subj in range(x_ch[tp].shape[0]):
                # if the mask for that subj is 0 at all features, continue
                if mask_ch[tp][subj].sum() == 0:
                    continue

                x_ch[tp][subj][mask_ch[tp][subj]] = torch.FloatTensor(x_ch_missing[subj][tp]).to(DEVICE)
    return X

# Paths and parameters 
model_dir = '/homedtic/gmarti/EXPERIMENTS_MCVAE/final_hyperparameter_search/_h_50_z_30_hid_50_n_0/'
model_path = f'{model_dir}model.pt'
out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/missingdata_results/"

seed = 1714

# Data paths
train_path = "data/multimodal_no_petfluid_train.csv"
test_csv = "data/multimodal_no_petfluid_test.csv"

# create output dir if doesnt exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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

if p["long_to_bl"]:
    # HERE, change bl to long and repeat the values at t0 for ntp
    for i in range(len(p["ch_type"])):
        if p["ch_type"][i] == 'bl':
            for j in range(len(X_train[i])):
                X_train[i][j] = np.array([X_train[i][j][0]]*ntp) 

            # p["ch_type"][i] = 'long'

X_train_list, mask_train_lsit = convert_X_to_tensor(X_train)

#####################
### TEST DATA ######
#####################
ch_bl_test = [] ##STORE THE CHANNELS THAT WE CONVERT TO LONG BUT WERE BL

X_test, _, Y_test, _, col_lists = load_multimodal_data(test_csv, channels, p["ch_type"], train_set=1.0, normalize=True, return_covariates=True)

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

X_test_list, mask_test_list = convert_X_to_tensor(X_test)

print('X_train')
#cada llista es un canal
# i es de size ntp, Nsubj, feats
print(len(X_train_list))
print(X_train_list[0].shape)

#Y train es un diccionary de 7 elements, i l'important és 'DX'
print('X_test')
print(len(X_test_list))
print(X_test_list[0].shape)

#####################
### MODEL ##########
#####################

# load the MCVAE model
# Now, load the model (the final version)
model = rnnvae_s.MCRNNVAE(p["h_size"], p["enc_hidden"],
                        p["enc_n_layers"], p["z_dim"], p["enc_hidden"], p["enc_n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], 
                        p["n_channels"], p["ch_type"], p["n_feats"], p["c_z"], DEVICE, print_every=100, 
                        phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                        dropout=p["dropout"], dropout_threshold=p["drop_th"])
model = model.to(DEVICE)
model.load(model_path)

# Create and save the GFA model for every channel combination
# can we save it to disk? and load from disk if it exists?
robjects.numpy2ri.activate()

# import R's utility package
utils = rpackages.importr('utils')
if not rpackages.isinstalled('GFA'):
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    utils.install_packages('GFA', lib="/homedtic/gmarti/R_libs/")
gfa_package=rpackages.importr('GFA', lib_loc="/homedtic/gmarti/R_libs/")

##Check if the gfa model is in disk, if it is, load it with pickle. If not, create it and save it
gfa_model_path = os.path.join(out_dir, 'gfa_model.pkl')
if os.path.exists(gfa_model_path):
    with open(gfa_model_path, 'rb') as f:
        gfa_model = pickle.load(f)
else:

    # PREPARE INPUT DATA
    # List of three matrices with the values,
    # remove temporal component and select only timepoints that are present at each channel
    Y = [[] for _ in range(len(X_train))]
    for i in range(len(X_train[0])):
        # select only the min len of each subject
        min_len = np.min([len(x[i]) for x in X_train])
        for j in range(len(X_train)):
            Y[j].append(X_train[j][i][:min_len])
            #Y[j].append(X_train[j][i][0])

    # prepare it arranging the timepoints
    Y = [np.array([tp for subj in Y_ch for tp in subj]) for Y_ch in Y]
    #Y = [np.array([tp for tp in Y_ch]) for Y_ch in Y]

    #NORMALIZE DATA USING THE PACKAGE
    opts = gfa_package.getDefaultOpts()
    tic = time.time()
    opts = gfa_package.informativeNoisePrior(Y, opts)
    gfa_model = gfa_package.gfa(Y, opts)
    toc = time.time()
    print("Time to compute GFA: ", toc-tic)

    # SAVE THE MODEL
    with open(gfa_model_path, 'wb') as f:
        pickle.dump(gfa_model, f)


#######################################################
################### MAIN LOOP #########################
#######################################################


# structure to save the results
# dictionary where the keys are in the format 'method_test'
# and each value is a list, with the same size as missing_perc,
#containing tuples in the format (mean, std) of the MAE, for all the trials
rec_results = {
    ## Reconstruct from all to this channel
    f'KNN_{channels[0]}': [],
    f'KNN_{channels[1]}': [],
    f'KNN_{channels[2]}': [],
    
    f'GFA_{channels[0]}': [],
    f'GFA_{channels[1]}': [],
    f'GFA_{channels[2]}': [],
    
    f'MCRVAE_{channels[0]}': [],
    f'MCRVAE_{channels[1]}': [],
    f'MCRVAE_{channels[2]}': [],

    ##Reconstruct channel to channel
    f'KNN_{channels[0]}_to_{channels[0]}': [],
    f'KNN_{channels[0]}_to_{channels[1]}': [],
    f'KNN_{channels[0]}_to_{channels[2]}': [],
    f'KNN_{channels[1]}_to_{channels[0]}': [],
    f'KNN_{channels[1]}_to_{channels[1]}': [],
    f'KNN_{channels[1]}_to_{channels[2]}': [],
    f'KNN_{channels[2]}_to_{channels[0]}': [],
    f'KNN_{channels[2]}_to_{channels[1]}': [],
    f'KNN_{channels[2]}_to_{channels[2]}': [],

    f'GFA_{channels[0]}_to_{channels[1]}': [],
    f'GFA_{channels[0]}_to_{channels[2]}': [],
    f'GFA_{channels[1]}_to_{channels[0]}': [],
    f'GFA_{channels[1]}_to_{channels[2]}': [],
    f'GFA_{channels[2]}_to_{channels[0]}': [],
    f'GFA_{channels[2]}_to_{channels[1]}': [],

    f'MCRVAE_{channels[0]}_to_{channels[0]}': [],
    f'MCRVAE_{channels[0]}_to_{channels[1]}': [],
    f'MCRVAE_{channels[0]}_to_{channels[2]}': [],
    f'MCRVAE_{channels[1]}_to_{channels[0]}': [],
    f'MCRVAE_{channels[1]}_to_{channels[1]}': [],
    f'MCRVAE_{channels[1]}_to_{channels[2]}': [],
    f'MCRVAE_{channels[2]}_to_{channels[0]}': [],
    f'MCRVAE_{channels[2]}_to_{channels[1]}': [],
    f'MCRVAE_{channels[2]}_to_{channels[2]}': []
}

# List of percentages of missing data, from 10 to 90, 0.05 increments
missing_perc = np.arange(0.0, 0.9, 0.1)

#  Make a copy of rec_results with the same structure
#  to save the results of each trial
rec_results_trial = copy.deepcopy(rec_results)

#for each percentage of missing data
for perc in missing_perc:

    # for each trial, make a copy of the original structure
    rec_results_perc = copy.deepcopy(rec_results_trial.copy)

    for i in range(5):

        # apply randomly to the test set
        X_test_missing = copy.deepcopy(X_test)
        X_test_missing = apply_missing_data(X_test_missing, perc)

        # And add the same exact missing data to the tensor X_test_list
        X_test_list_missing = mising_data_to_tensor(X_test_list, X_test_missing, mask_test_list)

        # PREPARE gFA MODEL for missing data reconstruction, as it is always the same (in fact, we can directly do the reconstruction)
        X_test_GFA, X_test_GFA_recon = prepare_gfa_recon_model(X_test, X_test_missing)

        #####
        # Test 2: Predict a modality from other modalities
        #####
        for j, ch in enumerate(channels):
            # Prepare the channel to predict
            y_true = X_test[j]
            y_true_GFA = X_test_GFA[j]
            # Reconstruct from all to this channel
            # KNN
            y_pred = knn_recon(X_train, X_test_missing, j, [0,1,2])
            rec_results_perc[f'KNN_{ch}'].append(my_mean_absolute_error(y_true, y_pred))
            print('KNN', ch)
            # GFA
            y_pred = gfa_recon(X_test_GFA_recon, gfa_model, j, [0,1,2])
            rec_results_perc[f'GFA_{ch}'].append(my_mean_absolute_error(y_true_GFA, y_pred, True))
            print('GFA', ch)
            # MCRVAE
            y_pred = mcrvae_recon(X_test_list_missing, mask_test_list, model, j, [0,1,2])
            rec_results_perc[f'MCRVAE_{ch}'].append(my_mean_absolute_error(y_true, y_pred))
            print('MCRVAE', ch)
            # Reconstruct channel to channel
            for k, ch2 in enumerate(channels):
                # KNN
                y_pred = knn_recon(X_train, X_test_missing, j, [k])
                rec_results_perc[f'KNN_{ch2}_to_{ch}'].append(my_mean_absolute_error(y_true, y_pred))
                print('KNN', ch2)
                # GFA
                # GFA doesnt work for the same channel
                if j != k:
                    y_pred = gfa_recon(X_test_GFA_recon, gfa_model, j, [k])
                    rec_results_perc[f'GFA_{ch2}_to_{ch}'].append(my_mean_absolute_error(y_true_GFA, y_pred, True))
                    print('GFA', ch2)
                # MCRVAE
                y_pred = mcrvae_recon(X_test_list_missing, mask_test_list, model, j, [k])
                rec_results_perc[f'MCRVAE_{ch2}_to_{ch}'].append(my_mean_absolute_error(y_true, y_pred))
                print('MCRVAE', ch2)

    #Compute mean and std of the results
    for key in rec_results_perc.keys():
        mean = np.mean([x[0] for x in rec_results_perc[key]])
        std = np.sqrt(np.mean([x[1]**2 for x in rec_results_perc[key]]))
        rec_results[key].append((mean, std))
    
#######################################################
################### SAVE THE RESULTS ##################
#######################################################
# Save the results in a csv file, converting the dictionary
# to a dataframe
df = pd.DataFrame.from_dict(rec_results)
df.to_csv(f'{out_dir}/results_full.csv')