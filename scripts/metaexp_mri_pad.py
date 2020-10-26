"""
Auxiliar file to run a lot of experiments on a single experiments

"""
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from test_mri_padding import run_experiment
from sklearn.model_selection import ParameterGrid
import configparser
import time
from datetime import timedelta
import pandas as pd

### Define grid search of parameters to look over for
params_grid = {
    "h_size": [2,5,10],
    "z_dim": [3,4,5],
    "hidden": [2,5,10],
    "n_layers": [1,2],
    "n_epochs": [2000],
    "clip": [10],
    "learning_rate": [1e-3],
    "batch_size": [128],
    "seed": [1714],
    "noise": [0.15],
    }

#Create two lists, that will store the dictionaries of the loss that later will become a dataframe
list_loss = []

csv_path = "data/tadpole_mrionly.csv"

base_out_dir = "experiments/meta_mri_lin_pad/"

if not os.path.exists(base_out_dir):
    os.makedirs(base_out_dir)

sys.stdout = open(base_out_dir + 'general_output.out', 'w')

#Run over the existing parameters
for p in ParameterGrid(params_grid):
    #decide out_dir DEPENDING ON what we are testing
    h_size = p['h_size']
    z_dim = p['z_dim']
    hidden = p['hidden']
    n_layers = p['n_layers']
    out_dir = f"{base_out_dir}_h_{h_size}_z_{z_dim}_hid_{hidden}_l_{n_layers}/"
    print("Running: " + out_dir)
    t = time.time()
    loss = run_experiment(p, csv_path, out_dir)
    sys.stdout = open(base_out_dir + 'general_output.out', 'w')
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