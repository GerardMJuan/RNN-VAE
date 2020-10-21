# Imports
import sys
sys.path.insert(0, '/homedtic/gmarti/CODE/RNN-VAE/')
from rnnvae.utils import open_MRI_data_var
from rnnvae import rnnvae
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_2d, plot_z_time_2d, plot_latent_space
import os
import math
from rnnvae.data_gen import SinDataGenerator
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import torch

# DEVICE
## Decidint on device on device.
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)

print(DEVICE)

out_dir = "experiments_mc/synth_nopadding/"

#load parameters
p = eval(open(out_dir + "params.txt").read())

model = rnnvae.MCRNNVAE(p["h_size"], p["hidden"], p["n_layers"], 
                        p["hidden"], p["n_layers"], p["hidden"],
                        p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], 
                        p["n_channels"], p["n_feats"], p["model_name_dict"], DEVICE)
model = model.to(DEVICE)
model.load(out_dir+'model.pt')

X_samples_tensor = []
nsamples = 500
#generate the data
for ch_curves in p['curves']:

    gen_model = SinDataGenerator(ch_curves, p["ntp"], p["noise"])
    samples = gen_model.generate_n_samples(nsamples)
    X_samples = np.asarray([y for (_,y) in samples])
    X_samples_tensor.append(torch.FloatTensor(X_samples).permute((1,0,2)).to(DEVICE))

# Predict the reconstructions from X_val and X_train
X_sample = model.predict(X_samples_tensor, p['ntp'])

#Get the samples
X_pred = [np.array(x).swapaxes(0,1) for x in X_sample['xnext']]
z_sample = [np.array(x).swapaxes(0,1) for x in X_sample['z']]

# Test the new function of latent space
qzx = [np.array(x) for x in X_sample['qzx']]
print(len(qzx))
print(len(qzx[0]))

# Get classificator labels, for n time points
classif = [[i]*nsamples for i in range(p["ntp"])]
classif = np.array([str(item) for elem in classif for item in elem])

out_dir_sample = out_dir + 'test_zspace_function/'
if not os.path.exists(out_dir_sample):
    os.makedirs(out_dir_sample)

plot_latent_space(model, qzx, p["ntp"], classificator=classif, plt_tp='all',
                  all_plots=False, uncertainty=True, savefig=True, out_dir=out_dir_sample)

plot_latent_space(model, qzx, p["ntp"], classificator=classif, plt_tp='all',
                  all_plots=True, uncertainty=True, savefig=True, out_dir=out_dir_sample)