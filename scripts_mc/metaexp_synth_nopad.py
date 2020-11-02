"""
Auxiliar file to run a lot of experiments on a single experiments

"""
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from test_synth_nopad import run_experiment
from sklearn.model_selection import ParameterGrid
import configparser
import time
from datetime import timedelta
import pandas as pd

curves = [
    [("sigmoid", {"L": 1, "k": 1, "x0": 5}),    
    ("sin", {"A": 1, "f": 0.2}),
    ("cos", {"A": 1, "f": 0.2})],
    [("sigmoid", {"L": 1, "k": -15, "x0": 5}),
    ("sigmoid", {"L": 1, "k": 5, "x0": 5})]
    ]

names = {"0":"c1", 
         "1":"c2"}

### Parameter definition
params_grid = {
    "h_size": [2,5,10],
    "z_dim": [2,3,4],
    "hidden": [2,5,10],
    "n_layers": [1],
    "n_epochs": [2000],
    "clip": [10],
    "learning_rate": [1e-3],
    "batch_size": [128],
    "seed": [1714],
    "curves": [curves],
    "ntp": [10,15],
    "noise": [0.15],
    "nsamples": [100,300,800],
    "n_channels": [len(curves)],
    "n_feats": [[len(x) for x in curves]],
    "model_name_dict": [names]
}

#Create two lists, that will store the dictionaries of the loss that later will become a dataframe
list_loss = []

base_out_dir = "experiments_mc/meta_synth_nopadding/"

if not os.path.exists(base_out_dir):
    os.makedirs(base_out_dir)

# sys.stdout = open(base_out_dir + 'general_output.out', 'w')

#Run over the existing parameters
for p in ParameterGrid(params_grid):
    #decide out_dir DEPENDING ON what we are testing
    h_size = p['h_size']
    z_dim = p['z_dim']
    hidden = p['hidden']
    ntp = p['ntp']
    nsamples = p['nsamples']
    out_dir = f"{base_out_dir}_h_{h_size}_z_{z_dim}_hid_{hidden}_ntp_{ntp}_n_{nsamples}/"
    print("Running: " + out_dir)
    t = time.time()
    loss = run_experiment(p, out_dir)
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