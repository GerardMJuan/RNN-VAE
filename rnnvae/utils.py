from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import pandas as pd

def pandas_to_data_timeseries(df, feat, n_timesteps = 5, id_col = 'PTID', time='VISCODE'):
    """
    Quick function that converts a pandas dataframe with the features
    indicated by a vector "feat" (with the name of the features columns) and "id_col"
    indicating the column of the subject ids, and column time to order them by time.

    We assume that the data is already preprocessed so that, for each PTID, there are n_timesteps
    rows.
    """
    # Order the dataframe
    df = df.sort_values(by=[id_col, time], ascending=False)

    #Nuumber of samples
    sample_list = np.unique(df[id_col])

    # Create base numpy structure
    X = np.zeros((len(sample_list), n_timesteps, len(feat)))
    # Iterate over each subject and fill it
    df_feats = df.loc[:, feat]
    i = 0
    for ptid in sample_list:
        i_list = df.index[df['PTID'] == ptid]
        feats = df_feats.iloc[i_list, :].values
        X[i, :, :] = feats
        i += 1

    # Return numpy dataframe
    return X

def pandas_to_data_timeseries_var(df, feat, id_col = 'PTID'):
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
    i = 0
    for ptid in sample_list:
        i_list = df.index[df['PTID'] == ptid]
        feats = df_feats.iloc[i_list, :].values
        X.append(feats)
        i += 1

    # Return numpy dataframe
    return X

def open_MRI_data(csv_path, train_set = 0.8, n_followups=5, normalize=True):
    """
    open MRI data from the specified directory
    We only return subjects with n_followups. If less, not included. If more, truncated.
    Divide between test and train.
    Return with the correct format (Nsamples, timesteps, nfeatures)
    (normalize parameter not used)

    NOTE: NEW VERSION SHOULD SORT THE DATA SEQUENCES FROM LONG TO SHORT, AND OUTPUTS ACCORDINGLY
    """

    data_df = pd.read_csv(csv_path)

    mri_col = data_df.columns.str.contains("SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16")
    mri_col = data_df.columns[mri_col].values

    data_df = data_df.dropna(axis=0, subset=mri_col)

    # Select only the subjects with nfollowups
    # Code to only select 5 first appearances of each PTID
    ptid_list = np.unique(data_df["PTID"])

    idx_to_drop = []
    for ptid in ptid_list:
        i_list = data_df.index[data_df['PTID'] == ptid].tolist()
        if len(i_list) < 5:
            idx_to_drop = idx_to_drop + i_list
        elif len(i_list) > 5:
            idx_to_drop = idx_to_drop + i_list[5:]

    data_final = data_df.drop(idx_to_drop)

    print(data_final.shape)

    # Normalize only features
    data_final.loc[:,mri_col] = data_final.loc[:,mri_col].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

    # Divide between test and train
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=1.0-train_set)
    train_dataset, test_dataset = next(gss.split(X=data_final, y=data_final.DX_bl.values, groups=data_final.PTID.values))

    df_train = data_final.iloc[train_dataset]
    df_test =  data_final.iloc[test_dataset]

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    X_train = pandas_to_data_timeseries(df_train, mri_col)
    X_test = pandas_to_data_timeseries(df_test, mri_col)

    if 

    return X_train, X_test


def open_MRI_data_var(csv_path, train_set = 0.8, normalize=True, return_covariates=False):
    """
    Function to return a variable number of followups from a dataset

    Returns:
    X_test: list composed of tensors of variable length
    X_train: list composed of tensors of variable length
    """
    data_df = pd.read_csv(csv_path)

    mri_col = data_df.columns.str.contains("SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16")
    mri_col = data_df.columns[mri_col].values

    data_df = data_df.dropna(axis=0, subset=mri_col)

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

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Return the features in the correct shape (Nsamples, timesteps, nfeatures)
    # Order the dataframes
    df_train = df_train.sort_values(by=['PTID', 'VISCODE'], ascending=False)
    df_test = df_test.sort_values(by=['PTID', 'VISCODE'], ascending=False)

    # Return the features in the correct shape list of Tensors (timesteps, nfeatures)
    X_train = pandas_to_data_timeseries_var(df_train, mri_col)
    X_test = pandas_to_data_timeseries_var(df_test, mri_col)

    # Columns which contains covariates
    # #TODO TODO TODO TODO    
    if return_covariates:
        cov_cols = []


    return X_train, X_test

