"""
Testing the full ADNI pipeline with the Multi Channel version

Load different channels and test it
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae_drop
from rnnvae.utils import load_multimodal_data
from rnnvae.plot import plot_losses, plot_trajectory, plot_total_loss, plot_z_time_2d, plot_latent_space
from sklearn.metrics import mean_squared_error
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

    # LOAD DATA
    #Start by not using validation data
    # this is a list of values
    X_train, X_test, Y_train, Y_test, mri_col = load_multimodal_data(csv_path, data_cols, train_set=0.8, normalize=True, return_covariates=True)

    p["n_feats"] = [x[0].shape[1] for x in X_train]

    X_train_list = []
    mask_train_list = []

    X_test_list = []
    mask_test_list = []

    print('Length of train/test')
    print(len(X_train[0]))
    # print(len(X_test[0]))

    #For each channel, pad, create the mask, and append
    for x_ch in X_train:
        X_train_tensor = [ torch.FloatTensor(t) for t in x_ch]
        X_train_pad = nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=False, padding_value=np.nan)
        mask_train = ~torch.isnan(X_train_pad)
        mask_train_list.append(mask_train.to(DEVICE))
        X_train_pad[torch.isnan(X_train_pad)] = 0
        X_train_list.append(X_train_pad.to(DEVICE))

    for x_ch in X_test:
        X_test_tensor = [ torch.FloatTensor(t) for t in x_ch]
        X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
        mask_test = ~torch.isnan(X_test_pad)
        mask_test_list.append(mask_test.to(DEVICE))
        X_test_pad[torch.isnan(X_test_pad)] = 0
        X_test_list.append(X_test_pad.to(DEVICE))

    ntp = max(X_train_list[0].shape[0], X_test_list[0].shape[0])

    model = rnnvae.MCRNNVAE(p["h_size"], p["hidden"], p["n_layers"], 
                            p["hidden"], p["n_layers"], p["hidden"],
                            p["n_layers"], p["z_dim"], p["hidden"], p["n_layers"],
                            p["clip"], p["n_epochs"], p["batch_size"], 
                            p["n_channels"], p["n_feats"], DEVICE)

    model.ch_name = p["ch_names"]

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    model.optimizer = optimizer

    model = model.to(DEVICE)
    # Fit the model
    model.fit(X_train_list, X_test_list, mask_train_list, mask_test_list)

    ### After training, save the model!
    model.save(out_dir, 'model.pt')

    # Predict the reconstructions from X_val and X_train
    X_train_fwd = model.predict(X_train_list, nt=ntp)
    X_test_fwd = model.predict(X_test_list, nt=ntp)

    # Unpad using the masks
    #plot validation and 
    plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir, "total_loss.png")
    plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir, "kl_loss.png")
    plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir, "ll_loss.png") #Negative to see downard curve

    #Compute mse and reconstruction loss
    #General mse and reconstruction over 
    # test_loss = model.recon_loss(X_test_fwd, target=X_test_pad, mask=mask_test_tensor)
    train_loss = model.recon_loss(X_train_fwd, target=X_train_list, mask=mask_train_list)
    test_loss = model.recon_loss(X_test_fwd, target=X_test_list, mask=mask_test_list)

    print('MSE over the train set: ' + str(train_loss["mae"]))
    print('Reconstruction loss over the train set: ' + str(train_loss["rec_loss"]))

    print('MSE over the test set: ' + str(test_loss["mae"]))
    print('Reconstruction loss the train set: ' + str(test_loss["rec_loss"]))

    ## Prediction of last time point

    # Test data without last timepoint

    # Run prediction

    #Compute MCVAE over last timepoint

    ## Test reconstruction over all channels.

    # For each channel

    # try to reconstruct it from the other ones

    # Get result


    # Dir for projections
    proj_path = 'z_proj/'
    if not os.path.exists(out_dir + proj_path):
        os.makedirs(out_dir + proj_path)

    # Test the new function of latent space
    #NEED TO ADAPT THIS FUNCTION
    qzx_train = [np.array(x) for x in X_train_fwd['qzx']]
    qzx_test = [np.array(x) for x in X_test_fwd['qzx']]

    #Convert to standard
    #Add padding so that the mask also works here
    DX_train = [[x for x in elem] for elem in Y_train["DX"]]
    DX_test = [[x for x in elem] for elem in Y_test["DX"]]

    #Define colors
    pallete_dict = {
        "CN": "#2a9e1e",
        "MCI": "#bfbc1a",
        "AD": "#af1f1f"
    }

    # Get classificator labels, for n time points
    out_dir_sample = out_dir + 'zcomp_ch_dx/'
    if not os.path.exists(out_dir_sample):
        os.makedirs(out_dir_sample)

    plot_latent_space(model, qzx_test, ntp, classificator=DX_test, pallete_dict=pallete_dict, plt_tp='all',
                all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)

    plot_latent_space(model, qzx_train, ntp, classificator=DX_train, pallete_dict=pallete_dict, plt_tp='all',
                    all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample + '_train', mask=mask_train_list)
    
    out_dir_sample_t0 = out_dir + 'zcomp_ch_dx_t0/'
    if not os.path.exists(out_dir_sample_t0):
        os.makedirs(out_dir_sample_t0)

    plot_latent_space(model, qzx_train, ntp, classificator=DX_train, pallete_dict=pallete_dict, plt_tp=[0],
                    all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample_t0 + '_train', mask=mask_train_list)

    plot_latent_space(model, qzx_test, ntp, classificator=DX_test, pallete_dict=pallete_dict, plt_tp=[0],
                    all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample_t0 + '_test', mask=mask_test_list)

    # Now plot color by timepoint
    out_dir_sample = out_dir + 'zcomp_ch_tp/'
    if not os.path.exists(out_dir_sample):
        os.makedirs(out_dir_sample)

    classif_train = [[i for (i, x) in enumerate(elem)] for elem in Y_train["DX"]]
    classif_test = [[i for (i, x) in enumerate(elem)] for elem in Y_test["DX"]]

    pallete = sns.color_palette("viridis", ntp)
    pallete_dict = {i:value for (i, value) in enumerate(pallete)}

    plot_latent_space(model, qzx_train, ntp, classificator=classif_train, pallete_dict=pallete_dict, plt_tp='all',
                    all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample + '_train', mask=mask_train_list)

    plot_latent_space(model, qzx_test, ntp, classificator=classif_test, pallete_dict=pallete_dict, plt_tp='all',
                    all_plots=True, uncertainty=False, savefig=True, out_dir=out_dir_sample + '_test', mask=mask_test_list)

    loss = {
        "mse_train" : train_loss["mae"],
        "rec_train" : train_loss["rec_loss"],
        "mse_test": test_loss["mae"],
        "loss_total": model.loss['total'][-1],
        "loss_kl": model.loss['kl'][-1],
        "loss_ll": model.loss['ll'][-1]
    }

    return loss

if __name__ == "__main__":

    ### Parameter definition

    channels = ['_mri_vol','_mri_cort','_demog','_apoe', '_cog', '_fluid','_fdg','_av45']
    names = ["MRI vol", "MRI cort", "Demog", "APOE", "Cog", "Fluid", "FDG", "AV45"]

    params = {
        "h_size": 32,
        "z_dim": 5,
        "hidden": 64,
        "n_layers": 1,
        "n_epochs": 20,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
        "n_channels": len(channels),
        "ch_names" : names
    }

    out_dir = "experiments_mc/MRI_ADNI_first/"
    csv_path = "data/full_multimodal.csv"
    loss = run_experiment(params, csv_path, out_dir, channels)