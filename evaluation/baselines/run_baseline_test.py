"""
Script to run the different baseline scores with different baseline methods

First method is a constant.

Second method is a linear regression individually for each subject.

Third method is a reconstruction based on k-means over the reconstruction channels.

Fourth method is a reconstruction method based on predicting the parameters of a linear model
based on the features of another dimension.

Basically: from a 

"""
# Imports
import sys
sys.path.insert(0, '/homedtic/gmarti/CODE/RNN-VAE/')
from rnnvae import rnnvae
from rnnvae.utils import load_multimodal_data, denormalize, denormalize_timepoint
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
from sklearn.tree import DecisionTreeRegressor
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


def predict_knn(X_train, X_test, n_ch, av_ch=None):
    """
    Predict missing modality (indicated by i) using a KNN approach for each modality. 
    the most similar one in the KNN space and averagin them to form the output
    we consider all timepoints as separate points for the knn
    later, to create the output, we just do the mean of all the channels, across all timepoints, even if they
    are variable
    """
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
        # if i == n_ch: 
        #     #Append nothing to keep ids consistents
        #     fitting_data.append([])
        #     X_missingmod_knn.append([])
        #     continue #dont create one for the original tal

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

            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(fitting_data[i])
            subj = X_ch[subj_idx]
            #for every timepoint
            for sample in subj:
                #find most similar subject in our training set
                idx = knn.kneighbors(sample.reshape(1, -1), return_distance=False)[0,0]
                #get the corresponding values of that modality
                pred_sample = X_missingmod_knn[i][idx]
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

    return y_pred


def create_lin_model_features(X):
    """
    Creates a linear model for each subject and return the mean and intercept 
    of each subject as features
    """
    X_ret = []
    for x in X:
        ntp = len(x)
        X_lin = np.array(range(ntp)).reshape(-1, 1) # X is the time points
        linreg_x = LinearRegression()
        linreg_x.fit(X_lin, x)
        X_ret.append(np.concatenate((linreg_x.coef_.squeeze(), linreg_x.intercept_)))

    return X_ret



def predict_rf(X_train, Y_train, X_test, Y_test_true):
    """
    Baseline to reconstruct one channel from another. Steps:
    1: Create a linear model for each X_train and Y_train sequence. Get a set of linear
        parameters for training and as output
    2: Train a random forest model with each separate sample of X_train associated
        to the corresponding two variables of the linear model (m and intercept)
    3: Use the trained model to predict trajectories of the Y_test (which we also need to obtain the linear
       parameters)
    4: Predict the points using the obtained parameters
    """
    # data structures to save the params for later training
    p_train_X = create_lin_model_features(X_train)
    p_train_Y = create_lin_model_features(Y_train)
    p_test_X = create_lin_model_features(X_test)

    # append coef and intercept and add to the X_train 
    # can append in any order right?
    tree = DecisionTreeRegressor()
    tree.fit(p_train_X, p_train_Y)
    
    #Get the trajectories
    Y_test = tree.predict(p_test_X)
    # predict, for each subject
    y_pred = []
    i = 0
    for subj in Y_test:
        coef = subj[:len(subj)//2]
        intercept = subj[len(subj)//2:]
        ntp = len(Y_test_true[i]) # Used to get the timepoints that we want
        X_lin = np.array(range(ntp)).reshape(-1, 1) 
        subj_pred = np.array([x*coef + intercept for x in X_lin])
        y_pred.append(subj_pred)
        i += 1
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
    for (X_ch, ch) in zip(X_test[:3], channels_train[:3]):
        #Select a single channel
        print(f'testing for {ch}')

        y_true = [x[-1] for x in X_ch if len(x) > 1]
        X_ch_train = [x[:-1] for x in X_ch if len(x) > 1]

        ### Constant
        y_pred_const = predict_constant(X_ch_train)
        # y_pred_const = [denormalize_timepoint(y, ch) for y in y_pred_const]
        constant_err = mean_absolute_error(y_true, y_pred_const)
        pred_results[f'test1_{ch}_constant'] = constant_err

        ### Linear
        y_pred_lin = predict_linear(X_ch_train)
        # y_pred_lin = [denormalize_timepoint(y, ch) for y in y_pred_lin]

        linear_err = mean_absolute_error(y_true, y_pred_lin)
        pred_results[f'test1_{ch}_linear'] = linear_err

    #####
    # Test 2: Predict a modality from other modalities
    #####
    print('Baseline for test 2: predicting channel from others')
    rec_results = {}
    i = 0 #channel to predict
    #TODO: iterate over channels
    for (X_ch, ch) in zip(X_test, channels_train):
        # Select a single channel as true gt
        print(f'testing for {ch}')
        y_true = X_ch

        # KNN
        # Predict that channel using all the others
        y_pred = predict_knn(X_train, X_test, i)

        # get only the length of y_pred that also appear in y_true
        y_pred_s = [x_pred[:min(len(x_true), len(x_pred))] for (x_pred, x_true) in zip(y_pred, y_true)]
        y_true_s = [x_true[:min(len(x_true), len(x_pred))] for (x_pred, x_true) in zip(y_pred, y_true)]

        #prepare it timepoint wise
        y_pred_s = [tp for subj in y_pred_s for tp in subj]
        y_true_s = [tp for subj in y_true_s for tp in subj]
        knn_err = mean_absolute_error(y_true_s, y_pred_s)
        rec_results[f"test2_knn_{ch}_"] = knn_err
        i += 1

    #####
    # Test 2.1: Predict a modality from other modalities
    #####
    results_knn = np.zeros((len(names_test), len(names_test))) #store the results, will save later

    # DONT INCLUDE APOE
    for i in range(len(names_test)):
        for j in range(len(names_test)):
            print(i," ",j)
            y_true = X_test[j]

            # predict it
            y_pred = predict_knn(X_train, X_test, j, [i])

            # get only the length of y_pred that also appear in y_true
            y_pred_s = [x_pred[:min(len(x_true), len(x_pred))] for (x_pred, x_true) in zip(y_pred, y_true)]
            y_true_s = [x_true[:min(len(x_true), len(x_pred))] for (x_pred, x_true) in zip(y_pred, y_true)]

            #prepare it timepoint wise
            y_pred_s = [tp for subj in y_pred_s for tp in subj]
            y_true_s = [tp for subj in y_true_s for tp in subj]
            knn_err = mean_absolute_error(y_true_s, y_pred_s)
            results_knn[i,j] = knn_err

    df_results = pd.DataFrame(data=results_knn, index=names_test, columns=names_test)
    plt.tight_layout()
    ax = sns.heatmap(df_results, annot=True, fmt=".2f")
    plt.savefig(out_dir + "figure_results_rnn.png")
    plt.close()

    #########
    # Test 3: Predict a modality from other modalities using the random forest predictor
    # across channels
    # only from channel to channel
    #########
    results_rf = np.zeros((len(names_test)-1, len(names_test)-1)) #store the results, will save later

    # DONT INCLUDE APOE
    for i in range(len(names_test)-1):
        for j in range(len(names_test)-1):
            print(i," ",j)
            y_true = X_test[j]

            # predict it
            y_pred = predict_rf(X_train[i], X_train[j], X_test[i], y_true)

            # get only the length of y_pred that also appear in y_true
            # y_pred = [x_pred[:len(x_true)] for (x_pred, x_true) in zip(y_pred, y_true)]

            #prepare it timepoint wise
            y_pred = [tp for subj in y_pred for tp in subj]
            y_true = [tp for subj in y_true for tp in subj]

            err = mean_absolute_error(y_true, y_pred)
            results_rf[i,j] = err

    df_results = pd.DataFrame(data=results_rf, index=names_test[:-1], columns=names_test[:-1])
    plt.tight_layout()
    ax = sns.heatmap(df_results, annot=True, fmt=".2f")
    plt.savefig(out_dir + "figure_results_rf.png")

    # SAVE AS FIGURE
    df_results.to_latex(out_dir+"table_crossrecon_rf.tex")

    #Save all the results on a csv file
    loss = {**pred_results, **rec_results}

    df_loss = pd.DataFrame([loss])

    for (test, val) in loss.items():
        print(f'{test}= {val}')
    #Order the dataframes
    df_loss.to_csv(out_dir + "baseline_loss_tp.csv")

if __name__ == "__main__":
    # Just hard code everything
    out_dir = "/homedtic/gmarti/CODE/RNN-VAE/evaluation/baselines/"
    in_csv_train = "data/multimodal_no_petfluid_train.csv"
    in_csv_test = "data/multimodal_no_petfluid_test.csv"
    main(in_csv_train, in_csv_test, out_dir)

    
