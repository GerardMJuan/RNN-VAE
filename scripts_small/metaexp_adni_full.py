"""
Metaexperiment for adni using baseline data and a CV procedure

This file trains the model several times over a 
"""
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from train_adni_full import run_experiment
from test_set_eval import run_eval
from sklearn.model_selection import ParameterGrid
import configparser
import time
from datetime import timedelta
import pandas as pd

channels = ['_mri_vol','_mri_cort', '_cog']#, '_demog', '_apoe']
names = ["MRI vol", "MRI cort", "Cog"]#, "Demog", 'APOE']
ch_type = ["long", "long", "long"]#, "bl", 'bl']

constrain1=[None, None, 5]#, 5, 5]

params = {
    "h_size": [30,50,70],
    "z_dim": [15,20,30],
    "x_hidden": [10],
    "x_n_layers": [1],
    "z_hidden": [15],
    "z_n_layers": [1],
    "enc_hidden": [20, 50, 80],
    "enc_n_layers": [0, 1, 2, 3],
    "dec_hidden": [10, 20],
    "dec_n_layers": [0],
    "n_epochs": [3500],
    "clip": [10],
    "learning_rate": [2e-3],
    "batch_size": [128],
    "seed": [1714],
    "c_z": [constrain1],
    "n_channels": [len(channels)],
    "ch_names" : [names],
    "ch_type": [ch_type],
    "phi_layers": [True],
    "sig_mean": [False],
    "dropout": [True],
    "drop_th": [0.2],
    "long_to_bl": [True]
}

csv_path = "data/multimodal_no_petfluid_train.csv"
csv_path_test = "data/multimodal_no_petfluid_test.csv"

#Create two lists, that will store the dictionaries of the loss that later will become a dataframe
list_loss = []
list_test = []

base_out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/revision_test_DELET/"

if not os.path.exists(base_out_dir):
    os.makedirs(base_out_dir)

# sys.stdout = open(base_out_dir + 'general_output.out', 'w')

#Run over the existing parameters
for p in ParameterGrid(params):
    #decide out_dir DEPENDING ON what we are testing
    h_size = p['h_size']
    z_dim = p['z_dim']
    hidden = p['enc_hidden']
    n_layers = p["enc_n_layers"]

    out_dir = f"{base_out_dir}_h_{h_size}_z_{z_dim}_hid_{hidden}_n_{n_layers}/"

    t = time.time()
    loss = run_experiment(p, csv_path, out_dir, channels, output_to_file=True)
    df_loss = pd.DataFrame([loss])
    df_loss.to_csv(out_dir + "cv_results.csv")    
    elapsed = time.time() - t
    
    # Merge dictionaries with params, so that runs can be identified
    # only relevant params
    rel_p = {
        "h_size": h_size,
        "z_dim": z_dim,
        "hidden": hidden,
        "n_layers": n_layers}
    #loss = {**loss, **rel_p}

    # Append to corresponding list
    #list_loss.append(loss)

    # Evaluate the results
    dropout=0.2
    type='val'
    res = run_eval(out_dir, csv_path, channels, dropout, type=type, output_to_file=True)

    # if test, need to save into a df or csv or similar so that I don't have to enter every time
    loss = res.to_dict()
    loss = {**loss, **rel_p}
    list_loss.append(loss)
    break

#Convert lists to dataframes
df_loss = pd.DataFrame(list_loss)

#Order the dataframes
#df_loss.sort_values(by="loss_total", inplace=True)

# Save them in out_dir
df_loss.to_csv(base_out_dir + "test_results.csv")