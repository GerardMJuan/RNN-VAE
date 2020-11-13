import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from random import randint
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

def pandas_to_data_timeseries_var(df, feat, normalize=True, id_col = 'PTID'):
    """
    Quick function that converts a pandas dataframe with the features
    indicated by a vector "feat" (with the name of the features columns) and "id_col"
    indicating the column of the subject ids, and column time to order them by time.

    The number of rows is variable, so we are creating a list of numpy arrays
    """
    #Nuumber of samples
    sample_list = np.unique(df[id_col])

    # Create base list
    X = []
    
    # Iterate over each subject and fill it
    df_feats = df.loc[:, feat]

    if normalize:
        # Standarize features
        for i in range(df_feats.shape[1]):
            df_feats.iloc[:,i] = (df_feats.iloc[:,i] - np.mean(df_feats.iloc[:,i]))/np.std(df_feats.iloc[:,i])

    for ptid in sample_list:
        i_list = df.index[df['PTID'] == ptid]
        # OPCIONAL, PER PROVAR LO DE PREDICCIÓ
        if len(i_list) < 2:
            continue
        feats = df_feats.iloc[i_list].values
        X.append(feats)

    # Return numpy dataframe
    return X


def pandas_to_data_timeseries(df, feat, n_timesteps = 5, normalize=True, id_col = 'PTID'):
    """
    Quick function that converts a pandas dataframe with the features
    indicated by a vector "feat" (with the name of the features columns) and "id_col"
    indicating the column of the subject ids, and column time to order them by time.

    We assume that the data is already preprocessed so that, for each PTID, there are n_timesteps
    rows.
    """

    #Nuumber of samples
    sample_list = np.unique(df[id_col])

    # Create base numpy structure
    X = []

    # Iterate over each subject and fill it
    df_feats = df.loc[:, feat]

    if normalize:
        # Standarize features
        for i in range(df_feats.shape[1]):
            df_feats.iloc[:,i] = (df_feats.iloc[:,i] - np.mean(df_feats.iloc[:,i]))/np.std(df_feats.iloc[:,i])

    for ptid in sample_list:
        i_list = df.index[df['PTID'] == ptid]
        feats = df_feats.iloc[i_list].values
        X.append(feats)

    # Return numpy dataframe
    return X

def open_MRI_data_var(csv_path, train_set = 0.8, normalize=True, return_covariates=False, data_cols='mri_vol'):
    """
    Function to return a variable number of followups from a dataset

    Returns:
    X_test: list composed of tensors of variable length
    X_train: list composed of tensors of variable length
    """
    data_df = pd.read_csv(csv_path)

    mri_col = data_df.columns.str.contains(data_cols)
    mri_col = data_df.columns[mri_col].values

    data_df = data_df.dropna(axis=0, subset=mri_col)

    #Drop columns where ptid do not have any bl
    data_df_bl = data_df[data_df.VISCODE == 'bl']       #select baselines
    ptid_with_bl = data_df_bl.PTID.unique()                #select which ptid have bl
    data_df = data_df[data_df.PTID.isin(ptid_with_bl)]  #remove the others

    # Select only the subjects with nfollowups
    # Code to only select 5 first appearances of each PTID
    ptid_list = np.unique(data_df["PTID"])

    idx_to_drop = []
    data_final = data_df.drop(idx_to_drop)

    # Divide between test and train
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=1.0-train_set)
    train_dataset, test_dataset = next(gss.split(X=data_final, y=data_final.DX_bl.values, groups=data_final.PTID.values))

    df_train = data_final.iloc[train_dataset]
    df_test =  data_final.iloc[test_dataset]

    # Return the features in the correct shape (Nsamples, timesteps, nfeatures)
    # Order the dataframes
    df_train = df_train.sort_values(by=['PTID', 'Years_bl'], ascending=[True, True])
    df_test = df_test.sort_values(by=['PTID', 'Years_bl'], ascending=[True, True])

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train.to_csv('train.csv')
    df_test.to_csv('test.csv')

    # Return the features in the correct shape list of Tensors (timesteps, nfeatures)
    X_train = pandas_to_data_timeseries_var(df_train, mri_col, normalize)
    X_test = pandas_to_data_timeseries_var(df_test, mri_col, normalize)

    # Columns which contains covariates
    if return_covariates:
        #No need but whatever
        df_y_train = df_train
        df_y_test = df_test

        # Add actual age columns
        df_y_train["AGE"] = df_y_train["AGE_demog"] + df_y_train["Years_bl"]
        df_y_test["AGE"] = df_y_test["AGE_demog"] + df_y_test["Years_bl"] 

        cov_cols = ["PTID", "AGE", "VISCODE","AGE_demog","PTGENDER_demog","PTEDUCAT_demog", "DX", "DX_bl", "Years_bl"]
        Y_train = {}
        Y_test = {}
        for col in cov_cols:
            Y_train[col] = pandas_to_data_timeseries_var(df_train, col, False)
            Y_test[col] = pandas_to_data_timeseries_var(df_test, col, False)

        return X_train, X_test, Y_train, Y_test, mri_col

    return X_train, X_test, mri_col


def open_MRI_data(csv_path, train_set = 0.8, n_followups=5, normalize=True, return_covariates=False):
    """
    open MRI data from the specified directory
    We only return subjects with n_followups. If less, not included. If more, truncated.
    Divide between test and train.
    Return with the correct format (Nsamples, timesteps, nfeatures)
    (normalize parameter not used)

    NOTE: NEW VERSION SHOULD SORT THE DATA SEQUENCES FROM LONG TO SHORT, AND OUTPUTS ACCORDINGLY
    """

    data_df = pd.read_csv(csv_path)

    mri_col = data_df.columns.str.contains("_mri_vol")
    mri_col = data_df.columns[mri_col].values

    data_df = data_df.dropna(axis=0, subset=mri_col)

    #Drop columns where ptid do not have any bl
    data_df_bl = data_df[data_df.VISCODE == 'bl']       #select baselines
    ptid_with_bl = data_df_bl.PTID.unique()                #select which ptid have bl
    data_df = data_df[data_df.PTID.isin(ptid_with_bl)]  #remove the others

    # Select only the subjects with nfollowups
    # Code to only select 5 first appearances of each PTID
    ptid_list = np.unique(data_df["PTID"])

    idx_to_drop = []
    for ptid in ptid_list:
        i_list = data_df.index[data_df['PTID'] == ptid].tolist()
        if len(i_list) < n_followups:
            idx_to_drop = idx_to_drop + i_list
        elif len(i_list) > n_followups:
            idx_to_drop = idx_to_drop + i_list[n_followups:]

    data_final = data_df.drop(idx_to_drop)

    print(data_final.shape)

    # Normalize only features
    # data_final.loc[:,mri_col] = data_final.loc[:,mri_col].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

    # Divide between test and train
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=1.0-train_set)
    train_dataset, test_dataset = next(gss.split(X=data_final, y=data_final.DX_bl.values, groups=data_final.PTID.values))

    df_train = data_final.iloc[train_dataset]
    df_test =  data_final.iloc[test_dataset]

    # Return the features in the correct shape (Nsamples, timesteps, nfeatures)
    # Order the dataframes
    df_train = df_train.sort_values(by=['PTID', 'Years_bl'], ascending=[True, True])
    df_test = df_test.sort_values(by=['PTID', 'Years_bl'], ascending=[True, True])

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    X_train = pandas_to_data_timeseries(df_train, mri_col, n_followups, normalize)
    X_test = pandas_to_data_timeseries(df_test, mri_col, n_followups, normalize)

    # Columns which contains covariates
    if return_covariates:
        #No need but whatever
        df_y_train = df_train
        df_y_test = df_test

        # Add actual age columns
        df_y_train["AGE"] = df_y_train["AGE_demog"] + df_y_train["Years_bl"]
        df_y_test["AGE"] = df_y_test["AGE_demog"] + df_y_test["Years_bl"] 


        cov_cols = ["PTID", "AGE", "VISCODE","AGE_demog","PTGENDER_demog","PTEDUCAT_demog", "DX", "DX_bl", "Years_bl"]
        Y_train = {}
        Y_test = {}
        for col in cov_cols:
            Y_train[col] = pandas_to_data_timeseries(df_y_train, col, n_followups, False)
            Y_test[col] = pandas_to_data_timeseries(df_y_test, col, n_followups, False)

        return X_train, X_test, Y_train, Y_test, mri_col


    return X_train, X_test, mri_col
