import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from random import randint
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
import pickle

def pickle_load(filename):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    return x

def pickle_dump(x, filename):
    with open(filename,'wb') as f:
        pickle.dump(x,f)


def denormalize_timepoint(x, suffix, norm_dict='data/norm_values/'):
    """
    Denormalize a single timepoint
    """
    norm_val = pickle.load( open(f"{norm_dict}{suffix}_norm.pkl", 'rb'))
    x_denorm = x * norm_val["std"] + norm_val["mean"]
    return x_denorm

def denormalize(X, suffixes=['mri_vol', 'mri_cort', 'cog', 'demog', 'apoe']):
    """
    Quick function to denormalize

    the data.
    X: input data, list of data channels
    suffixes: suffixes of each of the data channels to denormalize
    norm_dict: the directory where the normalizing values are
    """
    X_denorm = []
    for (x_ch, suffix) in zip(X, suffixes):
        #apply denorm over each timepoint and subject
        x_ch_denorm = [np.array([denormalize_timepoint(x_t, suffix) for x_t in x]) for x in x_ch]
        X_denorm.append(x_ch_denorm)

    return X_denorm


def pandas_to_data_timeseries_var(df, suffix, feat, normalize=True, id_col = 'PTID', norm_dict = 'data/norm_values/'):
    """
    Quick function that converts a pandas dataframe with the features
    indicated by a vector "feat" (with the name of the features columns) and "id_col"
    indicating the column of the subject ids, and column time to order them by time.
    The number of rows is variable, so we are creating a list of numpy arrays
    This is generalizable for all type sof features
    """
    #Nuumber of samples
    sample_list = np.unique(df[id_col])

    # Create base list
    X = []
    
    # Iterate over each subject and fill it
    df_feats = df.loc[:, feat]

    if normalize:
        #load the data  
        norm_val = pickle.load( open(f"{norm_dict}{suffix}_norm.pkl", 'rb'))
        df_feats = (df_feats - norm_val["mean"]) / norm_val["std"]

        # Standarize features
        #for i in range(df_feats.shape[1]):
        #    df_feats.iloc[:,i] = (df_feats.iloc[:,i] - np.mean(df_feats.iloc[:,i]))/np.std(df_feats.iloc[:,i])

    for ptid in sample_list:
        i_list = df.index[df['PTID'] == ptid]
        feats = df_feats.iloc[i_list].values
        X.append(feats)
    # Return numpy dataframe
    return X

def load_multimodal_data_cv(csv_path, suffixes_list, type_modal, nsplit=10, normalize=True):
    """
    This function returns several types of data from a csv dataset in different channels.
    Returns the data in the appropiate format, a list of different channels.
    if train_set=1.0, there is no divide between train and test
    channels is a list containing the suffixes of the columns of the original csv for that specific channel
    type_modal is a list with either "bl" or "long", depending on the type of data
    The data needs to already be preprocessed.
    """
    data_df = pd.read_csv(csv_path)

    dx_dict = {
        "NL": "CN",
        "MCI": "MCI",
        "MCI to NL": "CN",
        "Dementia": "AD",
        "Dementia to MCI": "MCI",
        "NL to MCI": "MCI",
        "NL to Dementia": "AD",
        "MCI to Dementia": "AD"
    }

    data_df['DX'] = data_df['DX'].map(dx_dict)
    data_df = data_df.sort_values(by=['PTID', 'Years_bl'], ascending=[True, True])

    gss = ShuffleSplit(n_splits=nsplit)
    for train_dataset, test_dataset in gss.split(X=data_df[data_df.VISCODE=='bl'].PTID.values, y=data_df[data_df.VISCODE=='bl'].DX.values):
        X_train_full = []
        X_test_full = []
        col_lists = []

        train_ptid = data_df[data_df.VISCODE=='bl'].PTID.values[train_dataset]
        test_ptid = data_df[data_df.VISCODE=='bl'].PTID.values[test_dataset]

        for (suffix, ch_type) in zip(suffixes_list, type_modal):

            cols = data_df.columns.str.contains(suffix)
            cols = data_df.columns[cols].values
            col_lists.append(cols)

            #Aquests linies NO haurien de fer falta perquè ja hem assegurat que tots els Bl TINGUIN tal.
            if ch_type == 'bl':
                data_df_base = data_df[data_df.VISCODE == 'bl']
            else:
                data_df_base = data_df.copy()
            data_df_base = data_df_base.dropna(axis=0, subset=cols)
            data_df_base = data_df_base.reset_index(drop=True)


            #Drop columns where ptid do not have any bl
            # data_df_bl = data_df[data_df.VISCODE == 'bl']       #select baselines
            # ptid_with_bl = data_df_bl.PTID.unique()                #select which ptid have bl
            # data_df = data_df[data_df.PTID.isin(ptid_with_bl)]  #remove the others

            # Select only the subjects with nfollowups
            # ptid_list = np.unique(data_df_base["PTID"])

            # Divide between test and train
            df_train = data_df_base[data_df_base.PTID.isin(train_ptid)]
            df_test = data_df_base[data_df_base.PTID.isin(test_ptid)]

            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)

            # df_train = data_df.iloc[train_dataset]
            #df_test =  data_df.iloc[test_dataset]

            # Return the features in the correct shape (Nsamples, timesteps, nfeatures)
            # Order the dataframes
            X_train = pandas_to_data_timeseries_var(df_train, suffix, cols, normalize)
            X_train_full.append(X_train)

            X_test = pandas_to_data_timeseries_var(df_test, suffix, cols, normalize)
            X_test_full.append(X_test)
        # Uncomment for debugging
        #df_train.to_csv('train.csv')
        #df_test.to_csv('test.csv')

        ## Covariates can be calculated using the last value of  df_train
        # Columns which contains covariates
        Y_train = {}
        Y_test = {}

        cov_cols = ["AGE_demog", "VISCODE","PTGENDER_demog","PTEDUCAT_demog", "DX", "DX_bl", "Years_bl"]

        # Divide between test and train
        df_train = data_df[data_df.PTID.isin(train_ptid)]
        df_test = data_df[data_df.PTID.isin(test_ptid)]

        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        for col in cov_cols:
            # no suffix, no normalization
            Y_train[col] = pandas_to_data_timeseries_var(df_train, None, col, False)
            Y_test[col] = pandas_to_data_timeseries_var(df_test, None, col, False)

        yield X_train_full, X_test_full, Y_train, Y_test, col_lists



def load_multimodal_data(csv_path, suffixes_list, type_modal, train_set=0.8, normalize=True, return_covariates=False):
    """
    This function returns several types of data from a csv dataset in different channels.
    Returns the data in the appropiate format, a list of different channels.

    if train_set=1.0, there is no divide between train and test
    channels is a list containing the suffixes of the columns of the original csv for that specific channel
    type_modal is a list with either "bl" or "long", depending on the type of data
    The data needs to already be preprocessed.

    """
    data_df = pd.read_csv(csv_path)

    dx_dict = {
        "NL": "CN",
        "MCI": "MCI",
        "MCI to NL": "CN",
        "Dementia": "AD",
        "Dementia to MCI": "MCI",
        "NL to MCI": "MCI",
        "NL to Dementia": "AD",
        "MCI to Dementia": "AD"
    }

    data_df['DX'] = data_df['DX'].map(dx_dict)
    data_df = data_df.sort_values(by=['PTID', 'Years_bl'], ascending=[True, True])

    test = True if train_set < 1.0 else False

    X_train_full = []
    X_test_full = []
    col_lists = []

    if test:
        from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
        gss = ShuffleSplit(n_splits=1, test_size=1.0-train_set)
        train_dataset, test_dataset = next(gss.split(X=data_df[data_df.VISCODE=='bl'].PTID.values, y=data_df[data_df.VISCODE=='bl'].DX.values))
        train_ptid = data_df[data_df.VISCODE=='bl'].PTID.values[train_dataset]
        test_ptid = data_df[data_df.VISCODE=='bl'].PTID.values[test_dataset]

    for (suffix, ch_type) in zip(suffixes_list, type_modal):

        cols = data_df.columns.str.contains(suffix)
        cols = data_df.columns[cols].values
        col_lists.append(cols)

        #Aquests linies NO haurien de fer falta perquè ja hem assegurat que tots els Bl TINGUIN tal.
        if ch_type == 'bl':
            data_df_base = data_df[data_df.VISCODE == 'bl']
        else:
            data_df_base = data_df.copy()
        data_df_base = data_df_base.dropna(axis=0, subset=cols)
        data_df_base = data_df_base.reset_index(drop=True)


        #Drop columns where ptid do not have any bl
        # data_df_bl = data_df[data_df.VISCODE == 'bl']       #select baselines
        # ptid_with_bl = data_df_bl.PTID.unique()                #select which ptid have bl
        # data_df = data_df[data_df.PTID.isin(ptid_with_bl)]  #remove the others

        # Select only the subjects with nfollowups
        ptid_list = np.unique(data_df_base["PTID"])

        if test:
            # Divide between test and train
            df_train = data_df_base[data_df_base.PTID.isin(train_ptid)]
            df_test = data_df_base[data_df_base.PTID.isin(test_ptid)]

            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)

            # df_train = data_df.iloc[train_dataset]
            #df_test =  data_df.iloc[test_dataset]
        else: 
            df_train = data_df_base.reset_index(drop=True)

        # Return the features in the correct shape (Nsamples, timesteps, nfeatures)
        # Order the dataframes
        X_train = pandas_to_data_timeseries_var(df_train, suffix, cols, normalize)
        X_train_full.append(X_train)

        if test: 
            X_test = pandas_to_data_timeseries_var(df_test, suffix, cols, normalize)
            X_test_full.append(X_test)
        # Uncomment for debugging
        #df_train.to_csv('train.csv')
        #df_test.to_csv('test.csv')

    ## Covariates can be calculated using the last value of  df_train
    # Columns which contains covariates
    if return_covariates:
        Y_train = {}
        Y_test = {}

        cov_cols = ["AGE_demog", "VISCODE","PTGENDER_demog","PTEDUCAT_demog", "DX", "DX_bl", "Years_bl"]

        if test:
            # Divide between test and train
            df_train = data_df[data_df.PTID.isin(train_ptid)]
            df_test = data_df[data_df.PTID.isin(test_ptid)]

            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)

        else:
            df_train = data_df.reset_index(drop=True)

        for col in cov_cols:
            Y_train[col] = pandas_to_data_timeseries_var(df_train, None, col, False)
            if test: Y_test[col] = pandas_to_data_timeseries_var(df_test, None, col, False)

        return X_train_full, X_test_full, Y_train, Y_test, col_lists

    return X_train_full, X_test_full, col_lists


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

def generate_norm_values(csv_path, suffixes_list, type_modal, out_dir):
    """
    Generate mean and std from the training set, for normalizing training set and test set alike.
    csv_path is where the information is stored.
    suffixes_list: list of suffixes of each modality. values will be saved together and with the same name + '_mean' and '_std'
    type_modal: is the type of modality, if bl or longitudinal.
    out_dir is where the outputs will be saved.
    """

    data_df = pd.read_csv(csv_path)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_df = data_df.sort_values(by=['PTID', 'Years_bl'], ascending=[True, True])

    for (suffix, ch_type) in zip(suffixes_list, type_modal):

        cols = data_df.columns.str.contains(suffix)
        cols = data_df.columns[cols].values

        #Aquests linies NO haurien de fer falta perquè ja hem assegurat que tots els Bl TINGUIN tal.
        if ch_type == 'bl':
            data_df_base = data_df[data_df.VISCODE == 'bl']
        else:
            data_df_base = data_df.copy()
        data_df_base = data_df_base.dropna(axis=0, subset=cols)
        data_df_base = data_df_base.reset_index(drop=True)

        # Select only the subjects with nfollowups
        df_train = data_df_base.reset_index(drop=True)

        #Generate mean and std of those features
        df_feats = df_train.loc[:, cols]

        # Generate the values
        mean = np.mean(df_feats).values
        std = np.std(df_feats).values
        #Save to disk
        norm_dict = {'mean' : mean, 'std' : std}

        with open(f"{out_dir}{suffix}_norm.pkl", 'wb') as f:
            pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)

def to_categorical_dataset(x,ch_name, normalized=True):
    """
    Auxiliar function to apply the "to categorical" function
    to all of the values of x.
    x is a list of subjects, and each subject has different timepoints
    """
    for s in range(len(x)):
        for tp in range(len(s)):
            x[s][tp] = to_categorical_individual(x[s][tp], ch_name, normalized)

    return x

def to_categorical_individual(x, ch_name, normalized=True):
    """
    Quick function to pass specific biomarkers to categorical.

    Cognitive, APOE, gender and years of education is inputted to the closest
    value in the categorical continuum.

    Need to gather those values from a previously generated file, dpeending on
    if values are normalized or not.

    x is a vector of size equal to the corresponding channel ch_name
    """
    data_path = "data/norm_values/"
    if normalized:
        filename = "categorical_norm"
    else:
        filename = "categorical_nonorm"
    dict_cat = pickle.load( open(f"{data_path}{filename}.pkl", 'rb'))

    if ch_name == "cog":
        #rounds to value that is closest
        x[0] = min(dict_cat["CDRSB_cog"], key=lambda k:abs(k-x[0]))
        x[1] = min(dict_cat["ADAS11_cog"], key=lambda k:abs(k-x[1]))
        x[2] = min(dict_cat["ADAS13_cog"], key=lambda k:abs(k-x[2]))
        x[3] = min(dict_cat["MMSE_cog"], key=lambda k:abs(k-x[3]))
        x[4] = min(dict_cat["RAVLT_immediate_cog"], key=lambda k:abs(k-x[4]))
        x[5] = min(dict_cat["FAQ_cog"], key=lambda k:abs(k-x[5]))
    elif ch_name == "demog":
        x[1] = min(dict_cat["PTGENDER_demog"], key=lambda k:abs(k-x[1]))
        x[2] = min(dict_cat["PTEDUCAT_demog"], key=lambda k:abs(k-x[2]))

    elif ch_name == "APOE":
        x[0] = min(dict_cat["APOE4_apoe"], key=lambda k:abs(k-x[0]))
    else:
        print("invalid channel name!")
    return x

    
    

