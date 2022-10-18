"""
Auxiliar file to run a lot of experiments on a single experiments

"""
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from test_synth import run_experiment
from sklearn.model_selection import ParameterGrid
from synth_eval import run_eval
import configparser
import time
from datetime import timedelta
import pandas as pd
from evaluate_synth import run_traj

names = ["c1","c2", "c3"]
ch_type = ["long", "long", "long"]

params_grid = {
    "h_size": [5, 10, 30],
    "z_dim": [20, 50, 100],
    "enc_hidden": [15,20,30],
    "enc_n_layers": [0, 1],
    "dec_hidden": [10],
    "dec_n_layers": [0],
    "n_epochs": [1000],
    "clip": [10],
    "learning_rate": [1e-2],
    "batch_size": [128],
    "seed": [1714],
    "ntp": [10],
    "noise": [1e-3],
    "nsamples": [1000],
    "n_channels": [3],
    "n_feats": [10],
    "lat_dim": [3],
    "c_z": [[None, None, None]],
    "ch_type": [ch_type],
    "ch_names" : [names],
    "phi_layers": [True],
    "sig_mean": [False],
    "dropout": [True],
    "drop_th": [0.2],
}

#Create two lists, that will store the dictionaries of the loss that later will become a dataframe
list_loss = []

base_out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/final_hyperparameter_synth/"

if not os.path.exists(base_out_dir):
    os.makedirs(base_out_dir)

#Run over the existing parameters
for p in ParameterGrid(params_grid):
    #decide out_dir DEPENDING ON what we are testing
    h_size = p['h_size']
    z_dim = p['z_dim']
    out_dir = f"{base_out_dir}_h_{h_size}_z_{z_dim}/"
    print("Running: " + out_dir)
    t = time.time()
    loss = run_experiment(p, out_dir, gen_data=False, data_suffix='_5', output_to_file=False)

    elapsed = time.time() - t
    print('Time to run: %s' % str(timedelta(seconds=elapsed)))
    # Merge dictionaries with params, so that runs can be identified
    # only relevant params
    rel_p = {
        "h_size": h_size,
        "z_dim": z_dim}

    loss = {**loss, **rel_p}

    # Append to corresponding list
    list_loss.append(loss)

    # Evaluate the results
    dropout=1.0
    run_eval(out_dir, names, dropout, output_to_file=False)

    #Plot the curves
    run_traj(out_dir, names, dropout, output_to_file=False)


#Convert lists to dataframes
df_loss = pd.DataFrame(list_loss)

#Order the dataframes
df_loss.sort_values(by="loss_total", inplace=True)

# Save them in out_dir
df_loss.to_csv(base_out_dir + "metaexperiment_loss.csv")