"""
Script to run the different baseline scores with different baseline methods
"""
# Imports
import sys
sys.path.insert(0, '/homedtic/gmarti/CODE/RNN-VAE/')
from rnnvae import rnnvae
from rnnvae.utils import load_multimodal_data
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from itertools import zip_longest

def predict_constant(X_train):
    """
    Function to predict next value just using the previous one
    """
    y_pred = [x[-1] for x in X_train]
    return y_pred


def predict_linear(X_train):
    """
    Predict next value using the training values of each 
    subject.
    """

    y_pred = []
    for Y in X_train:
        ntp = len(Y)
        X_lin = np.array(range(ntp)).reshape(-1, 1) # X is the time points
        linreg = LinearRegression()
        linreg.fit(X_lin, Y)
        Y_lin_pred = linreg.predict(np.array([ntp+1]).reshape(-1, 1)).flatten()
        y_pred.append(Y_lin_pred)

    return y_pred


def predict_knn(X_train, X_test, n_ch):
    """
    Predict missing modality (indicated by i) using a KNN approach for each modality. 
    the most similar one in the KNN space and averagin them to form the output
    we consider all timepoints as separate points for the knn
    later, to create the output, we just do the mean of all the channels, across all timepoints, even if they
    are variable
    """
    y_pred = []
    X_missing_mod = X_train[n_ch]
    # needs to be same length as training_knn

    fitting_data = []
    X_missingmod_knn = []
    #Fit knn for each modality
    for (i, X_ch) in enumerate(X_train):
        if i == n_ch: 
            #Append nothing to keep ids consistents
            fitting_data.append([])
            X_missingmod_knn.append([])
            continue #dont create one for the original tal
        print(i)

        training_knn = []
        corresponding_mod_knn = []
        #get data
        # only select 
        #for each subject, only select timepoints that also appear in the original data
        for (nsubj, x) in enumerate(X_ch):
            for (tp, sample) in enumerate(x):
                if tp < len(X_missing_mod[nsubj]):
                    training_knn.append(sample)
                    corresponding_mod_knn.append(X_missing_mod[nsubj][tp])

        fitting_data.append(training_knn)
        X_missingmod_knn.append(corresponding_mod_knn)

    # for every subject
    for subj_idx in range(len(X_test[0])):
        
        predicted_ch = [[] for _ in range(len(X_test))]
        # for every channel
        for (i, X_ch) in enumerate(X_test):
            if i == n_ch: continue # dont compute it over the missing modality

            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(fitting_data[i])
            subj = X_ch[subj_idx]
            #for every timepoint
            for sample in subj:
                #find most similar subject in our training set
                idx = knn.kneighbors(sample.reshape(1, -1), return_distance=False)[0,0]
                #get the corresponding values of that modality
                pred_sample = X_missingmod_knn[i][idx]
                predicted_ch[i].append(pred_sample)

        #Now, need to compute the mean across the samples
        subj_pred = list(zip_longest(*predicted_ch))
        # Compute the mean across every list (removing the Nones first)
        subj_pred = [[x for x in l if x is not None] for l in subj_pred]
        subj = [np.mean(p, axis=0) for p in subj_pred]
        y_pred.append(subj)

    return y_pred


def main(in_csv_train, in_csv_test, out_dir):
    """
    Main function to perform the experiments
    
    Loads the data from in_csv
    """
    ## Load the train data
    channels_train = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
    names_train = ["MRI vol", "MRI cort", "Cog", "Demog", 'APOE']
    ch_type_train = ["long", "long", "long", "bl", 'bl']
    X_train, Y_train, cols = load_multimodal_data(in_csv_train, channels_train, ch_type_train, train_set=1.0, normalize=True, return_covariates=False)

    ## Load the test data
    channels_test = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
    names_test = ["MRI vol", "MRI cort", "Cog","Demog", 'APOE']
    ch_type_test = ["long", "long", "long", "long", 'long']
    X_test, Y_test, cols = load_multimodal_data(in_csv_test, channels_test, ch_type_test, train_set=1.0, normalize=True, return_covariates=False)

    ########
    # Test 1: Predict next time point
    ########
    print('Baseline for test 1: predicting next time-point')
    pred_results = {}
    for (X_ch, ch) in zip(X_test[:3], names_test[:3]):
        #Select a single channel
        print(f'testing for {ch}')

        y_true = [x[-1] for x in X_ch if len(x) > 1]
        X_ch_train = [x[:-1] for x in X_ch if len(x) > 1]

        ### Constant
        y_pred_const = predict_constant(X_ch_train)
        constant_err = mean_absolute_error(y_true, y_pred_const)
        pred_results[f'test1_{ch}_constant'] = constant_err

        ### Linear
        y_pred_lin = predict_linear(X_ch_train)
        linear_err = mean_absolute_error(y_true, y_pred_lin)
        pred_results[f'test1_{ch}_linear'] = linear_err

    #####
    # Test 2: Predict a modality from other modalities
    #####
    print('Baseline for test 2: predicting channel from others')
    rec_results = {}
    i = 0 #channel to predict
    #TODO: iterate over channels
    for (X_ch, ch) in zip(X_test, names_test):
        # Select a single channel as true gt
        print(f'testing for {ch}')
        y_true = X_ch

        # KNN
        # Predict that channel using all the others
        y_pred = predict_knn(X_train, X_test, i)
        # get only the length of y_pred that also appear in y_true
        y_pred = [x_pred[:len(x_true)] for (x_pred, x_true) in zip(y_pred, y_true)]

        #prepare it timepoint wise
        y_pred = [tp for subj in y_pred for tp in subj]
        y_true = [tp for subj in y_true for tp in subj]
        knn_err = mean_absolute_error(y_true, y_pred)
        rec_results[f"test2_knn_{ch}_"] = knn_err
        i += 1

    #Save all the results on a csv file

    loss = {**pred_results, **rec_results}

    df_loss = pd.DataFrame([loss])

    #Order the dataframes
    df_loss.to_csv(out_dir + "baseline_loss_tp1.csv")

if __name__ == "__main__":
    # Just hard code everything
    out_dir = "/homedtic/gmarti/CODE/RNN-VAE/evaluation/baselines/"
    in_csv_train = "data/multimodal_no_petfluid_train.csv"
    in_csv_test = "data/multimodal_no_petfluid_test.csv"
    main(in_csv_train, in_csv_test, out_dir)

    
