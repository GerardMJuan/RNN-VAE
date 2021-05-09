"""
Script with functions to visualize the synthetic data, both the latent space and the trajectories of
the actual data.

Would be cooler in a jupyter script, but I am tired of using that on the cluster and how slow it is.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import random
import numpy as np
from rnnvae import rnnvae_s
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rnnvae.data_gen import LatentTemporalGenerator

def visualize_latent_space(Z, out_dir, feat=[0, 1], subj='all'):
    """
    Visualize two dimensions of the latent space, for all subjects (or some of them)
    and colored by some options
    Z: Latent space
    out_dir: output directory for the figures
    feat: list of two elements, default [0,1]
    subj: lists of subjects to plot, default, if all, select all
    """
    if subj=='all':
        subj = np.arange(len(Z))

    out_name = out_dir + f'dim{feat[0]}_{feat[1]}.png'

    # create the 2 dim matrix between those dimensions, and plot every subject over it
    f, ax = plt.subplots()
    for s in subj:
        x = [x_i[feat[0]] for x_i in Z[s]] # first dim
        y = [x_i[feat[1]] for x_i in Z[s]] # second dim
        sns.scatterplot(x=x, y=y, ax=ax)

    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()

def visualize_trajectory(X, out_dir, X_hat=None, feat='all', subj='all'):
    """
    Visualize the trajectory of a a subject (or all subjects), also can be compared to 
    prediction/reconstruction of a specific subject if needed
    X: data to plot. Shape
    out_dir: output directory of the figure
    X_hat: optional, prediction of X
    feat: which feat to plot, can be a list. If 'all', creates a figure for each feat
    subj: which subj to plot, can be a list. If 'all', creates a figure for each subj 
    """
    if feat == 'all':
        feat = np.arange(X[0][0].shape[1])
    if subj == 'all':
        subj = np.arange(len(X[0]))

    #for each channel, feat and subj, plot
    for ch in range(len(X)):
        for f in feat:
            for s in subj:
                x_i = X[ch][s][:,f]
                namefig = f'ch{ch}_s{s}_f{f}.png'
                plt.figure()
                sns.lineplot(x=x_i, y=np.arange(len(x_i)))
                # TODO: IF X_HAT, ADD ANOTHER LINEPLOT HERE WITH THE X INFORMATION
                plt.tight_layout()
                plt.savefig(out_dir + namefig)
                plt.close()

if __name__ == "__main__":
    """
    Main function.

    I put it directly here because fuck it I don't really care anymore.
    """
    
    out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/synth_testing/"

    #load parameters
    p = eval(open(out_dir + "params.txt").read())

    #Seed
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])

    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    #Tensors should have the shape
    # [ntp n_ch, n_batch, n_feat]
    #as n_feat can be different across channels, ntp and n_ch need to be lists. n_batch and n_feat are the tensors
    lat_gen = LatentTemporalGenerator(p["ntp"], p["noise"], p["lat_dim"], p["n_channels"], p["n_feats"])
    Z, X = lat_gen.generate_samples(p["nsamples"])

    # VISUALIZE LATENT SPACE, all dims
    out_dir_latent = out_dir + 'synth_latent/'
    if not os.path.exists(out_dir_latent):
        os.makedirs(out_dir_latent)

    for i in range(p["lat_dim"]):
        for j in range(i+1, p["lat_dim"]):
            visualize_latent_space(Z, out_dir_latent, feat=[i, j])

    # VISUALIZE TRAJECTORY, TEN RANDOM SUBJECTS AND F= 0
    subj = np.random.choice(np.arange(p["nsamples"]), 10, replace=False)
    out_dir_traj = out_dir + 'synth_trajectories/'
    if not os.path.exists(out_dir_traj):
        os.makedirs(out_dir_traj)
    visualize_trajectory(X, out_dir_traj, feat=[0], subj=subj)


