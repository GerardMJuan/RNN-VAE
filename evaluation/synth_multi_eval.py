"""
Script to evaluate synthetic data.

Generate multi-channel synthetic data, train a basic model, and check it against a baseline to predict it. 
Also test the dropout.
"""
# Imports
import sys
import os
sys.path.insert(0, '/homedtic/gmarti/CODE/RNN-VAE/')
import math
from rnnvae.data_gen import LatentDataGeneratorCurves
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from rnnvae import rnnvae

# DEVICE
## Decidint on device on device.
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)


# data parameters
ntp = 10
noise = 0.1
variable_tp=False
lat_dim=2
n_channels=3
n_feats=20
n_samples=100
curves = [
    ("sigmoid", {"L": 1, "k": 1, "x0": 5}),    
    ("sin", {"A": 1, "f": 0.2}),
    ("cos", {"A": 1, "f": 0.2})
    ]
ch_type = ["long", "long", "long"]
# model parameters
p = {
    "h_size": 20,
    "z_dim": 5,
    "hidden": 20,
    "n_layers": 1,
    "n_epochs": 6000,
    "clip": 10,
    "learning_rate": 5e-3,
    "batch_size": 128,
    "seed": 1714,
    "n_channels": n_channels,
    "ch_type": ch_type,
    "phi_layers": False,
    "sig_mean": False,
    "dropout": True,
    "drop_th": 0.2,
    "n_feats": [n_feats for _ in range(n_channels)]
}

out_dir = "/homedtic/gmarti/RNN-VAE/test_synth_mc/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

lat_gen = LatentDataGeneratorCurves(curves, ch_type, ntp, noise, lat_dim, n_channels, n_feats)
Z, X = lat_gen.generate_samples(n_samples)

"""
plt.figure()
for z_t in Z:
    plt.scatter(z_t[:,0], z_t[:,1], c='blue', marker='o')
nsubj = 5
for z_t in Z:
    plt.scatter(z_t[nsubj,0], z_t[nsubj,1], c='red', marker='x')
print(out_dir + 'testing_z_latent_space.png')
plt.show()
plt.savefig(out_dir + 'testing_z_latent_space.png')
"""
###
# SELECT VARIABLE TIME POINTS FOR EACH TAL
###

X_train_list = []
mask_train_list = []

#generate the data, and the mask corresponding to each channel
for x_ch in X:
    #originally, x_ch is ntp, nfeat, nsamples
    #  should be size nsamples, ntp, nfeat
    x_ch = x_ch.swapaxes(0,2).swapaxes(1,2)
    X_train_tensor = [ torch.FloatTensor(t) for t in x_ch ]
    X_train_pad_i = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False, padding_value=np.nan)
    mask_train = ~torch.isnan(X_train_pad_i)
    mask_train_list.append(mask_train.to(DEVICE))
    X_train_pad_i[torch.isnan(X_train_pad_i)] = 0
    X_train_list.append(X_train_pad_i.to(DEVICE))


model = rnnvae.MCRNNVAE(p["h_size"], p["hidden"], p["n_layers"], 
                        p["hidden"], p["n_layers"], p["hidden"],
                        p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                        p["clip"], p["n_epochs"], p["batch_size"], 
                        p["n_channels"], p["ch_type"], p["n_feats"], DEVICE, print_every=100, 
                        phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                        dropout=p["dropout"], dropout_threshold=p["drop_th"])

optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
model.optimizer = optimizer

model = model.to(DEVICE)
# Fit the model
model.fit(X_train_list, X_train_list, mask_train_list, mask_train_list)
### After training, save the model!
model.save(out_dir, 'model.pt')

if p["dropout"]:
    print("Print the dropout")
    print(model.dropout_comp)

# Predict the reconstructions from X_val and X_train
X_train_fwd = model.predict(X_train_list, nt=ntp)

# Unpad using the masks
#plot validation and tal
plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir, "total_loss.png")
plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir, "kl_loss.png")
plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir, "ll_loss.png") #Negative to see downard curve


###Plot the dropout, sorted