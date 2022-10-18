"""
Auxiliar file to run a lot of experiments on a single file

"""
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from train_adni_full import run_experiment
from sklearn.model_selection import ParameterGrid
import configparser
import time
from datetime import timedelta
import pandas as pd

channels = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
names = ["MRI vol", "MRI cort", "Cog", "Demog", 'APOE']
ch_type = ["long", "long", "long", "bl", 'bl']

constrain1=[None, None, 5, 5, 5]
constrain2=[None, None, 2, 2, 2]


params = {
    "h_size": [50,75,100],
    "z_dim": [15,20,30],
    "phi_x_hidden": [10],
    "phi_x_n_layers": [1],
    "phi_z_hidden": [20,30],
    "phi_z_n_layers": [1],
    "enc_hidden": [120],
    "enc_n_layers": [1],
    "dec_hidden": [120],
    "dec_n_layers": [1],
    "n_epochs": [2000],
    "clip": [10],
    "learning_rate": [1e-3],
    "batch_size": [128],
    "seed": [1714],
    "c_z": [constrain1, constrain2],
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

#Create two lists, that will store the dictionaries of the loss that later will become a dataframe
list_loss = []

base_out_dir = "/homedtic/gmarti/EXPERIMENTS/RNNVAE/metaexp_FULLNetwork/"

if not os.path.exists(base_out_dir):
    os.makedirs(base_out_dir)

# sys.stdout = open(base_out_dir + 'general_output.out', 'w')

#Run over the existing parameters
for p in ParameterGrid(params):
    #decide out_dir DEPENDING ON what we are testing
    h_size = p['h_size']
    z_dim = p['z_dim']
    z_hidden = p['phi_z_hidden']
    c_z = p["c_z"]

    out_dir = f"{base_out_dir}_h_{h_size}_z_{z_dim}_zhid_{z_hidden}_cz_{c_z[3]}/"

    t = time.time()
    loss = run_experiment(p, csv_path, out_dir, channels)
    df_loss = pd.DataFrame([loss])
    df_loss.to_csv(out_dir + "cv_results.csv")    
    elapsed = time.time() - t
    # Merge dictionaries with params, so that runs can be identified
    # only relevant params
    rel_p = {
        "h_size": h_size,
        "z_dim": z_dim,
        "z_hidden": z_hidden,
        "constraint_z": c_z[3]}
    loss = {**loss, **rel_p}

    # Append to corresponding list
    list_loss.append(loss)


#Convert lists to dataframes
df_loss = pd.DataFrame(list_loss)

#Order the dataframes
#df_loss.sort_values(by="loss_total", inplace=True)

# Save them in out_dir
df_loss.to_csv(base_out_dir + "cv_results.csv")