"""
Auxiliar file to run a lot of experiments on a single file

"""
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from test_adni_val_full import run_experiment
from sklearn.model_selection import ParameterGrid
import configparser
import time
from datetime import timedelta
import pandas as pd

channels = ['_mri_vol','_mri_cort', '_cog']
names = ["MRI vol", "MRI cort", "Cog"]
ch_type = ["long", "long", "long"]

params = {
    "h_size": [5,10],
    "z_dim": [50,100],
    "hidden": [20],
    "n_layers": [0,1],
    "n_epochs": [5000],
    "clip": [10],
    "learning_rate": [1e-3,1e-4],
    "batch_size": [128],
    "seed": [1714],
    "n_channels": [len(channels)],
    "ch_names" : [names],
    "ch_type": [ch_type],
    "phi_layers": [True],
    "sig_mean": [False],
    "dropout": [True],
    "drop_th": [0.3]
}

csv_path = "data/multimodal_no_petfluid.csv"

#Create two lists, that will store the dictionaries of the loss that later will become a dataframe
list_loss = []

base_out_dir = "experiments_mc_newloss/dropout_testing/"

if not os.path.exists(base_out_dir):
    os.makedirs(base_out_dir)

# sys.stdout = open(base_out_dir + 'general_output.out', 'w')

#Run over the existing parameters
for p in ParameterGrid(params):
    #decide out_dir DEPENDING ON what we are testing
    h_size = p['h_size']
    z_dim = p['z_dim']
    hidden = p['hidden']
    nlayers = p["n_layers"]
    learning_rate = p["learning_rate"]
    out_dir = f"{base_out_dir}_h_{h_size}_z_{z_dim}_hid_{hidden}_nl_{nlayers}_lr_{learning_rate}/"
    #out_dir = f"{base_out_dir}_h_{h_size}_z_{z_dim}_lr_{learning_rate}/"
    print("Running: " + out_dir)
    t = time.time()
    loss = run_experiment(p, csv_path, out_dir, channels)
    #sys.stdout = open(base_out_dir + 'general_output.out', 'w')
    elapsed = time.time() - t
    print('Time to run: %s' % str(timedelta(seconds=elapsed)))
    # Merge dictionaries with params, so that runs can be identified
    loss = {**loss, **p}

    # Append to corresponding list
    list_loss.append(loss)

#Convert lists to dataframes
df_loss = pd.DataFrame(list_loss)

#Order the dataframes
df_loss.sort_values(by="loss_total", inplace=True)

# Save them in out_dir
df_loss.to_csv(base_out_dir + "metaexperiment_loss.csv")