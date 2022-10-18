"""
test_mc_synth.py

Testing for a single channel for synthetic,
longitudinal data. This file will be used. We will generate
different length signals to input to our model and see how it behaves.

Also, two different settings:
Test with padding and different lengths. We use the new masking thing.
Its also variable ACROSS CHANNELS, 
"""

#imports
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae_h
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_many_trajectories, plot_latent_space
from rnnvae.data_gen import LatentDataGeneratorCurves


def run_experiment(p, out_dir):
    """
    Function to run the experiments.
    p contain all the hyperparameters needed to run the experiments
    We assume that all the parameters needed are present in p!!
    out_dir is the out directory
    #hyperparameters
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #Seed
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])

    #Redirect output to the out dir
    # sys.stdout = open(out_dir + 'output.out', 'w')

    #save parameters to the out dir 
    with open(out_dir + "params.txt","w") as f:
        f.write(str(p))
    
    # DEVICE
    ## Decidint on device on device.
    DEVICE_ID = 0
    DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE_ID)

    #Tensors should have the shape
    # [ntp n_ch, n_batch, n_feat]
    #as n_feat can be different across channels, ntp and n_ch need to be lists. n_batch and n_feat are the tensors

    lat_gen = LatentDataGeneratorCurves(p['curves'], p["ch_type"], p["ntp"], p["noise"], p["lat_dim"], p["n_channels"], p["n_feats"])
    Z, X = lat_gen.generate_samples(p["nsamples"])


    X_train_list = []
    mask_train_list = []

    #generate the data, and the mask corresponding to each channel
    for x_ch in X:
        #originally, x_ch is ntp, nfeat, nsamples
        #  should be size nsamples, ntp, nfeat
        # import pdb; pdb.set_trace()
        # x_ch = x_ch.swapaxes(0,2).swapaxes(1,2)
        X_train_tensor = [ torch.FloatTensor(t) for t in x_ch ]
        X_train_pad_i = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False, padding_value=np.nan)
        mask_train = ~torch.isnan(X_train_pad_i)
        mask_train_list.append(mask_train.to(DEVICE))
        X_train_pad_i[torch.isnan(X_train_pad_i)] = 0
        X_train_list.append(X_train_pad_i.to(DEVICE))

    #Stack along first dimension
    #cant do that bc last dimension (features) could be different length
    # X_train_tensor = torch.stack(X_train_tensor, dim=0)
    #X_test_tensor = torch.stack(X_test_tensor, dim=0)
    p["n_feats"] = [p["n_feats"] for _ in range(p["n_channels"])]
    #Prepare model
    # Define model and optimizer
    model = rnnvae_h.MCRNNVAE(p["h_size"], p["x_hidden"], p["x_n_layers"], 
                            p["z_hidden"], p["z_n_layers"], p["enc_hidden"],
                            p["enc_n_layers"], p["z_dim"], p["dec_hidden"], p["dec_n_layers"],
                            p["clip"], p["n_epochs"], p["batch_size"], 
                            p["n_channels"], p["ch_type"], p["n_feats"], p["c_z"], DEVICE, print_every=100, 
                            phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                            dropout=p["dropout"], dropout_threshold=p["drop_th"])
    model.ch_name = p["ch_names"]

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    model.optimizer = optimizer

    model = model.to(DEVICE)
    # Fit the model
    model.fit(X_train_list, X_train_list, mask_train_list, mask_train_list)

    if p["dropout"]:
        print("Print the dropout")
        print(model.dropout_comp)


    ### After training, save the model!
    model.save(out_dir, 'model.pt')

    # Predict the reconstructions from X_val and X_train
    X_train_fwd = model.predict(X_train_list, mask_train_list, nt=p["ntp"])

    # Unpad using the masks
    #plot validation and 
    plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir, "total_loss.png")
    plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir, "kl_loss.png")
    plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir, "ll_loss.png") #Negative to see downard curve

    #Compute mse and reconstruction loss
    #General mse and reconstruction over 
    train_loss = model.recon_loss(X_train_fwd, target=X_train_list, mask=mask_train_list)

    print('MSE over the train set: ' + str(train_loss["mae"]))
    print('Reconstruction loss over the train set: ' + str(train_loss["rec_loss"]))

    ##Latent spasce
    #Reformulate things
    z_train = [np.array(x).swapaxes(0,1) for x in X_train_fwd['z']]

    # Dir for projections
    proj_path = 'z_proj/'
    if not os.path.exists(out_dir + proj_path):
        os.makedirs(out_dir + proj_path)

    #plot latent space
    for ch in range(p["n_channels"]):
        for dim0 in range(p["z_dim"]):
            for dim1 in range(dim0, p["z_dim"]):
                if dim0 == dim1: continue   # very dirty
                plot_z_time_2d(z_train[ch], p["ntp"], [dim0, dim1], out_dir + proj_path, out_name=f'z_ch_{ch}_d{dim0}_d{dim1}')

    # Dir for projections
    sampling_path = 'z_proj_sampling/'
    if not os.path.exists(out_dir + sampling_path):
        os.makedirs(out_dir + sampling_path)

    # Test the new function of latent space
    qzx = [np.array(x) for x in X_train_fwd['qzx']]

    # Get classificator labels, for n time points
    classif = [[i]*p["nsamples"] for i in range(p["ntp"])]
    classif = np.array([str(item) for elem in classif for item in elem])
    print("on_classif")
    print(p["ntp"])
    print(len(classif))

    out_dir_sample = out_dir + 'test_zspace_function/'
    if not os.path.exists(out_dir_sample):
        os.makedirs(out_dir_sample)

    # TODO: ADAPT THIS FUNCTION TO THE 
    #plot_latent_space(model, qzx, p["ntp"], classificator=classif, plt_tp='all',
    #                all_plots=False, uncertainty=True, savefig=True, out_dir=out_dir_sample)

    loss = {
        "mse_train" : train_loss["mae"],
        "loss_total": model.loss['total'][-1],
        "loss_kl": model.loss['kl'][-1],
        "loss_ll": model.loss['ll'][-1]
    }

    return loss

if __name__ == "__main__":

    curves = [
        ("sigmoid", {"L": 1, "k": 10, "x0": 5}),
        ("sigmoid", {"L": 1, "k": -5, "x0": 3}),
        ("sigmoid", {"L": 1, "k": -15, "x0": 1})
        ]

    names = ["c1","c2", "c3"]
    ch_type = ["long", "long", "long"]
    ### Parameter definition
    params = {
        "h_size": 10,
        "z_dim": 5,
        "x_hidden": 30,
        "x_n_layers": 1,
        "z_hidden": 20,
        "z_n_layers": 1,
        "enc_hidden": 120,
        "enc_n_layers": 0,
        "dec_hidden": 120,
        "dec_n_layers": 0,
        "n_epochs": 100,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
        "curves": curves,
        "ntp": 10,
        "noise": 0.15,
        "nsamples": 800,
        "n_channels": len(curves),
        "n_feats": 15,
        "lat_dim": 4,
        "c_z": [None, None, None],
        "ch_names" : names,
        "ch_type": ch_type,
        "phi_layers": True,
        "sig_mean": False,
        "dropout": True,
        "drop_th": 0.2,
        "long_to_bl": False
    }

    out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/synth_testing/"
    loss = run_experiment(params, out_dir)