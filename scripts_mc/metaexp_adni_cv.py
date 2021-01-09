"""
Metaexperiment for adni using baseline data and a CV procedure
"""
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from test_adni_cv import run_experiment
from sklearn.model_selection import ParameterGrid
import configparser
import time
from datetime import timedelta
import pandas as pd

channels = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
names = ["MRI vol", "MRI cort", "Cog", "Demog", 'APOE']
ch_type = ["long", "long", "long", "bl", 'bl']

params = {
    "h_size": [300,500],
    "z_dim": [20,30,50],
    "hidden": [300,500],
    "n_layers": [1,2],
    "n_epochs": [1300],
    "clip": [10],
    "learning_rate": [1e-3],
    "batch_size": [128],
    "seed": [1714],
    "n_channels": [len(channels)],
    "ch_names" : [names],
    "ch_type": [ch_type],
    "phi_layers": [True],
    "sig_mean": [False],
    "dropout": [False],
    "drop_th": [0.3]
}

csv_path = "data/multimodal_no_petfluid_train.csv"

#Create two lists, that will store the dictionaries of the loss that later will become a dataframe
list_loss = []

base_out_dir = "experiments_mc_cv/results_cv_full_2/"

if not os.path.exists(base_out_dir):
    os.makedirs(base_out_dir)

sys.stdout = open(base_out_dir + 'general_output.out', 'w')

#Run over the existing parameters
for p in ParameterGrid(params):
    #decide out_dir DEPENDING ON what we are testing
    h_size = p['h_size']
    z_dim = p['z_dim']
    hidden = p['hidden']
    nlayers = p["n_layers"]
    learning_rate = p["learning_rate"]
    phi_layers = p["phi_layers"]
    dropout = p["dropout"]
    nepochs = p["n_epochs"]
    out_dir = f"{base_out_dir}_h_{h_size}_z_{z_dim}_hid_{hidden}_nl_{nlayers}_lr_{learning_rate}" +\
              f"_phi_{phi_layers}_drop_{dropout}_epochs_{nepochs}/"

    t = time.time()
    try:
        loss = run_experiment(p, csv_path, out_dir, channels)
        df_loss = pd.DataFrame([loss])
        df_loss.to_csv(out_dir + "cv_results.csv")    
        elapsed = time.time() - t
        # Merge dictionaries with params, so that runs can be identified
        # only relevant params
        rel_p = {
            "h_size": h_size,
            "z_dim": z_dim,
            "hidden": hidden,
            "nlayers": nlayers,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "phi_layers": phi_layers,
            "nepochs": nepochs}
        loss = {**loss, **rel_p}

        # Append to corresponding list
        list_loss.append(loss)
    except:
        print('it failed, continue')
        

#Convert lists to dataframes
df_loss = pd.DataFrame(list_loss)

#Order the dataframes
#df_loss.sort_values(by="loss_total", inplace=True)

# Save them in out_dir
df_loss.to_csv(base_out_dir + "cv_results.csv")