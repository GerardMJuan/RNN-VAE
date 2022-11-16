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
from rnnvae import rnnvae_s
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_many_trajectories, plot_latent_space
from rnnvae.data_gen import LatentTemporalGenerator
from rnnvae.utils import pickle_load, pickle_dump

def run_experiment(p, out_dir, gen_data=True, data_suffix=None, output_to_file=False):
    """
    Function to run the experiments.
    p contain all the hyperparameters needed to run the experiments
    We assume that all the parameters needed are present in p!!
    out_dir is the out directory
    gen_data: bool that indicates if we have to generate the data or load it from disk
    data_suffix: if gen_data=True, put the suffix of the data here, which will be in data/synth_data/ by defualt
    #hyperparameters
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #Seed
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])

    #Redirect output to the out dir
    if output_to_file:
        sys.stdout = open(out_dir + 'output.out', 'w')

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
    if gen_data:
        synth_dir = 'data/synth_data/'
        Z_train = pickle_load(synth_dir + f"ztrain{data_suffix}")
        Z_test = pickle_load(synth_dir + f"ztest{data_suffix}")
        X_train = pickle_load(synth_dir + f"xtrain{data_suffix}")
        X_test = pickle_load(synth_dir + f"xtest{data_suffix}")
    else:
        lat_gen = LatentTemporalGenerator(p["ntp"], p["noise"], p["lat_dim"], p["n_channels"], p["n_feats"])
        Z_train, X_train = lat_gen.generate_samples(p["nsamples"])
        Z_test, X_test = lat_gen.generate_samples(int(p["nsamples"]*0.2), train=False)

    # Save the data used to the output disk
    import pdb; pdb.set_trace()
    to_save = [Z_train, X_train, Z_test, X_test]
    filenames = [f"ztrain", f"xtrain", f"ztest", f"xtest"]
    for object, file in zip(to_save,filenames):
        pickle_dump(object, out_dir + file)

    X_train_list = []
    mask_train_list = []

    #generate the data, and the mask corresponding to each channel
    for x_ch in X_train[:p["n_channels"]]:
        X_train_tensor = [ torch.FloatTensor(t) for t in x_ch ]
        X_train_pad_i = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False, padding_value=np.nan)
        mask_train = ~torch.isnan(X_train_pad_i)
        mask_train_list.append(mask_train.to(DEVICE))
        X_train_pad_i[torch.isnan(X_train_pad_i)] = 0
        X_train_list.append(X_train_pad_i.to(DEVICE))

    X_test_list = []
    mask_test_list = []

    for x_ch in X_test[:p["n_channels"]]:
        X_test_tensor = [ torch.FloatTensor(t) for t in x_ch]
        X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
        mask_test = ~torch.isnan(X_test_pad)
        mask_test_list.append(mask_test.to(DEVICE))
        X_test_pad[torch.isnan(X_test_pad)] = 0
        X_test_list.append(X_test_pad.to(DEVICE))


    #Stack along first dimension
    #cant do that bc last dimension (features) could be different length
    p["n_feats"] = [p["n_feats"] for _ in range(p["n_channels"])]
    
    # Prepare model
    # Define model and optimizer
    model = rnnvae_s.MCRNNVAE(p["h_size"], p["enc_hidden"],
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
    model.fit(X_train_list, X_test_list, mask_train_list, mask_test_list)

    if p["dropout"]:
        print("Print the dropout")
        print(model.dropout_comp)

    ### After training, save the model!
    model.save(out_dir, 'model.pt')

    # Predict the reconstructions from X_val and X_train
    X_test_list = model.predict(X_test_list, mask_test_list, nt=p["ntp"])
    X_train_fwd = model.predict(X_train_list, mask_train_list, nt=p["ntp"])

    # Unpad using the masks
    #plot validation and 
    plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir, "total_loss.png")
    plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir, "kl_loss.png")
    plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir, "ll_loss.png") #Negative to see downard curve

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
        "loss_total": model.loss['total'][-1],
        "loss_kl": model.loss['kl'][-1],
        "loss_ll": model.loss['ll'][-1]
    }

    return loss

if __name__ == "__main__":

    names = ["c1","c2", "c3"]
    ch_type = ["long", "long", "long"]#, 'bl']#, "bl", 'bl']
    ### Parameter definition
    params = {
        "h_size": 50,
        "z_dim": 30,
        "enc_hidden": 120,
        "enc_n_layers": 0,
        "dec_hidden": 120,
        "dec_n_layers": 0,
        "n_epochs": 4000,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
        "ntp": 10,
        "noise": 1e-3,
        "nsamples": 800,
        "n_channels": 2,
        "n_feats": 20,
        "lat_dim": 5,
        "c_z": [None, None, None],
        "ch_type": ch_type,
        "ch_names" : names,
        "phi_layers": True,
        "sig_mean": False,
        "dropout": True,
        "drop_th": 0.2,
    }

    out_dir = "/homedtic/gmarti/EXPERIMENTS_MCVAE/synth_testing/"
    loss = run_experiment(params, out_dir, True, "_5", True)
    print(loss)