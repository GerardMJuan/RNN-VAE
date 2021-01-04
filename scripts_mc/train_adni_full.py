"""
Script with all the new parameters added, used for testing those
parameters.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae_h
from rnnvae.utils import load_multimodal_data
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

def run_experiment(p, csv_path, out_dir, data_cols=[]):
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

    # LOAD DATA
    #Start by not using validation data
    # this is a list of values
    X_train, _, Y_train, _, mri_col = load_multimodal_data(csv_path, data_cols, p["ch_type"], train_set=0.9, normalize=True, return_covariates=True)

    p["n_feats"] = [x[0].shape[1] for x in X_train]

    X_train_list = []
    mask_train_list = []


    print('Length of train/test')
    print(len(X_train[0]))

    #For each channel, pad, create the mask, and append
    for x_ch in X_train:
        X_train_tensor = [ torch.FloatTensor(t) for t in x_ch]
        X_train_pad = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False, padding_value=np.nan)
        mask_train = ~torch.isnan(X_train_pad)
        mask_train_list.append(mask_train.to(DEVICE))
        X_train_pad[torch.isnan(X_train_pad)] = 0
        X_train_list.append(X_train_pad.to(DEVICE))

    ntp = max(max([x.shape[0] for x in X_train_list]), max([x.shape[0] for x in X_train_list]))

    model = rnnvae_h.MCRNNVAE(p["h_size"], p["hidden"], p["n_layers"], 
                            p["hidden"], p["n_layers"], p["hidden"],
                            p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                            p["clip"], p["n_epochs"], p["batch_size"], 
                            p["n_channels"], p["ch_type"], p["n_feats"], DEVICE, print_every=100, 
                            phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                            dropout=p["dropout"], dropout_threshold=p["drop_th"])

    model.ch_name = p["ch_names"]

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    model.optimizer = optimizer

    model = model.to(DEVICE)
    # Fit the model
    model.fit(X_train_list, X_train_list, mask_train_list, mask_train_list)

    #fit the model after changing the lr
    #optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"]*.1)
    #model.optimizer = optimizer
    #print('Refining optimization...')
    #model.fit(X_train_list, X_test_list, mask_train_list, mask_test_list)
    if p["dropout"]:
        print("Print the dropout")
        print(model.dropout_comp)

    ### After training, save the model!
    model.save(out_dir, 'model.pt')

    # Predict the reconstructions from X_val and X_train
    X_train_fwd = model.predict(X_train_list, mask_train_list, nt=ntp)

    # Unpad using the masks
    #plot validation and 
    plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir, "total_loss.png")
    plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir, "kl_loss.png")
    plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir, "ll_loss.png") #Negative to see downard curve

    #Compute mse and reconstruction loss
    #General mse and reconstruction over 
    # test_loss = model.recon_loss(X_test_fwd, target=X_test_pad, mask=mask_test_tensor)
    train_loss = model.recon_loss(X_train_fwd, target=X_train_list, mask=mask_train_list)

    print('MSE over the train set: ' + str(train_loss["mae"]))
    print('Reconstruction loss over the train set: ' + str(train_loss["rec_loss"]))


    loss = {
        "mae_train" : train_loss["mae"],
        "rec_train" : train_loss["rec_loss"],
        "loss_total": model.loss['total'][-1],
        "loss_kl": model.loss['kl'][-1],
        "loss_ll": model.loss['ll'][-1],
    }

    if p["dropout"]:
        loss["dropout_comps"] = model.dropout_comp

    print(loss)

    return loss

if __name__ == "__main__":

    channels = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
    names = ["MRI vol", "MRI cort", "Cog", "Demog", 'APOE']
    ch_type = ["long", "long", "long", "bl", 'bl']

    params = {
        "h_size": 300,
        "z_dim": 20,
        "hidden": 300,
        "n_layers": 1,
        "n_epochs": 1500,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
        "n_channels": len(channels),
        "ch_names" : names,
        "ch_type": ch_type,
        "phi_layers": True,
        "sig_mean": False,
        "dropout": False,
        "drop_th": 0.3
    }

    out_dir = "experiments_mc_models/allch_fulltrain1500/"
    csv_path = "data/multimodal_no_petfluid_train.csv"
    loss = run_experiment(params, csv_path, out_dir, channels)
