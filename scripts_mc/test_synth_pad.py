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
from rnnvae import rnnvae
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_many_trajectories, plot_latent_space
from rnnvae.data_gen import SinDataGenerator


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
    X_train_pad = []
    X_test_pad = []

    mask_train_tensor = []
    mask_test_tensor = []

    #generate the data, and the mask corresponding to each channel
    for ch_curves in p['curves']:
        gen_model = SinDataGenerator(ch_curves, p["ntp"], p["noise"], variable_tp=True)
        samples = gen_model.generate_n_samples(p["nsamples"])
        X_train = np.asarray([y for (_,y) in samples])
        X_train_tensor = [ torch.FloatTensor(t) for t in X_train ]
        X_train_pad_i = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False, padding_value=np.nan)
        mask_train = ~torch.isnan(X_train_pad_i)
        mask_train_tensor.append(mask_train.to(DEVICE))
        X_train_pad_i[torch.isnan(X_train_pad_i)] = 0
        X_train_pad.append(X_train_pad_i.to(DEVICE))

        #Do teh same for the testing set
        samples = gen_model.generate_n_samples(int(p["nsamples"]*0.8))
        X_test = np.asarray([y for (_,y) in samples])
        X_test_tensor = [ torch.FloatTensor(t) for t in X_test ]
        X_test_pad_i = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
        mask_test = ~torch.isnan(X_test_pad_i)
        mask_test_tensor.append(mask_test.to(DEVICE))
        X_test_pad_i[torch.isnan(X_test_pad_i)] = 0
        X_test_pad.append(X_test_pad_i.to(DEVICE))

    #Stack along first dimension
    #cant do that bc last dimension (features) could be different length
    # X_train_tensor = torch.stack(X_train_tensor, dim=0)
    #X_test_tensor = torch.stack(X_test_tensor, dim=0)

    #Prepare model
    # Define model and optimizer
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
    model.fit(X_train_pad, X_test_pad, mask_train_tensor, mask_test_tensor)

    ### After training, save the model!
    model.save(out_dir, 'model.pt')

    # Predict the reconstructions from X_val and X_train
    X_test_fwd = model.predict(X_test_pad, nt=p["ntp"])
    X_train_fwd = model.predict(X_train_pad, nt=p["ntp"])

    # Unpad using the masks
    #plot validation and 
    plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir, "total_loss.png")
    plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir, "kl_loss.png")
    plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir, "ll_loss.png") #Negative to see downard curve

    #Compute mse and reconstruction loss
    #General mse and reconstruction over 
    test_loss = model.recon_loss(X_test_fwd, target=X_test_pad, mask=mask_test_tensor)
    train_loss = model.recon_loss(X_train_fwd, target=X_train_pad, mask=mask_train_tensor)

    print('MSE over the train set: ' + str(train_loss["mae"]))
    print('Reconstruction loss over the train set: ' + str(train_loss["rec_loss"]))

    print('MSE over the test set: ' + str(test_loss["mae"]))
    print('Reconstruction loss the train set: ' + str(test_loss["rec_loss"]))

    ##Latent spasce
    #Reformulate things
    z_train = [np.array(x).swapaxes(0,1) for x in X_train_fwd['z']]
    z_test = [np.array(x).swapaxes(0,1) for x in X_test_fwd['z']]

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
        "mse_test": test_loss["mae"],
        "loss_total": model.loss['total'][-1],
        "loss_kl": model.loss['kl'][-1],
        "loss_ll": model.loss['ll'][-1]
    }

    return loss

if __name__ == "__main__":

    curves = [
        [("sin", {"A": 1, "f": 0.5}),    
        ("sigmoid", {"L": 2, "k": -15, "x0": 5}),
        ("cos", {"A": 1, "f": 0.2})],
        [("sin", {"A": 1, "f": 0.2}),
        ("sigmoid", {"L": -2, "k": 5, "x0": 5})]
        ]
    
    names = {"0":"c1", 
             "1":"c2"}

    ch_type = ["long", "long"]

    ### Parameter definition
    params = {
        "h_size": 20,
        "z_dim": 10,
        "hidden": 20,
        "n_layers": 1,
        "n_epochs": 500,
        "clip": 10,
        "learning_rate": 1e-2,
        "batch_size": 128,
        "seed": 1714,
        "curves": curves,
        "ntp": 10,
        "noise": 0.1,
        "nsamples": 300,
        "n_channels": len(curves),
        "n_feats": [len(x) for x in curves],
        "model_name_dict": names,
        "ch_type": ch_type,
        "phi_layers": True,
        "sig_mean": False,
        "dropout": False,
        "drop_th": 0.4
    }

    out_dir = "experiments_mc_newloss/synth_padding/"
    loss = run_experiment(params, out_dir)