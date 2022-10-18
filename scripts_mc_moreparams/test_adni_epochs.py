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
from rnnvae.eval import eval_reconstruction, eval_prediction
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
    X_train, X_test, Y_train, Y_test, mri_col = load_multimodal_data(csv_path, data_cols, p["ch_type"], train_set=0.9, normalize=True, return_covariates=True)

    p["n_feats"] = [x[0].shape[1] for x in X_train]

    X_train_list = []
    mask_train_list = []

    X_test_list = []
    mask_test_list = []

    print('Length of train/test')
    print(len(X_train[0]))
    print(len(X_test[0]))

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

    # ntp = max(X_train_list[0].shape[0], X_test_list[0].shape[0])
    ntp = max(max([x.shape[0] for x in X_train_list]), max([x.shape[0] for x in X_train_list]))

    model = rnnvae_h.MCRNNVAE(p["h_size"], p["x_hidden"], p["x_n_layers"], 
                            p["z_hidden"], p["z_n_layers"], p["enc_hidden"],
                            p["enc_n_layers"], p["z_dim"], p["dec_hidden"], p["dec_n_layers"],
                            p["clip"], p["n_epochs"], p["batch_size"], 
                            p["n_channels"], p["ch_type"], p["n_feats"], DEVICE, print_every=100, 
                            phi_layers=p["phi_layers"], sigmoid_mean=p["sig_mean"],
                            dropout=p["dropout"], dropout_threshold=p["drop_th"])
                            
    model.ch_name = p["ch_names"]

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    model.optimizer = optimizer

    model = model.to(DEVICE)
    
    # Fit the model
    # FIT IT FOR THE NUMBER OF EPOCHS, X TIMES
    ntimes = 20
    for nrep in range(ntimes):
        print(nrep)
        model.fit(X_train_list, X_test_list, mask_train_list, mask_test_list)

        #fit the model after changing the lr
        if p["dropout"]:
            print("Print the dropout")
            print(model.dropout_comp)

        ### After training, save the model!
        model.save(out_dir, 'model.pt')

        # Predict the reconstructions from X_val and X_train
        X_train_fwd = model.predict(X_train_list, mask_train_list, nt=ntp)
        X_test_fwd = model.predict(X_test_list, mask_test_list, nt=ntp)

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

        pred_results = {}
        for ch_name in p["ch_names"][:3]:
            pred_results[f"pred_{ch_name}_mae"] = []

        rec_results = {}
        for ch_name in p["ch_names"]:
            rec_results[f"recon_{ch_name}_mae"] = []

        results = {**pred_results, **rec_results}

        ######################
        ## Prediction of last time point
        ######################

        i = 0
        # Test data without last timepoint
        # X_test_tensors do have the last timepoint
        pred_ch = list(range(3))
        print(pred_ch)
        t_pred = 1
        res = eval_prediction(model, X_test, t_pred, pred_ch, DEVICE)

        for (i,ch) in enumerate([x for (i,x) in enumerate(p["ch_names"]) if i in pred_ch]):
            loss[f'pred_{ch}_mae'].append(res[i])

        ############################
        ## Test reconstruction for each channel, using the other one 
        ############################
        # For each channel
        if p["n_channels"] > 1:

            for i in range(len(X_test)):
                curr_name = p["ch_names"][i]
                av_ch = list(range(len(X_test)))
                av_ch.remove(i)
                mae_rec = eval_reconstruction(model, X_test, X_test_list, mask_test_list, av_ch, i)
                # Get MAE result for that specific channel over all timepoints
                results[f"recon_{curr_name}_mae"] = mae_rec


        loss = {
            "mae_train" : train_loss["mae"],
            "rec_train" : train_loss["rec_loss"],
            "mae_test": test_loss["mae"],
            "loss_total": model.loss['total'][-1],
            "loss_kl": model.loss['kl'][-1],
            "loss_ll": model.loss['ll'][-1],
        }

        if p["dropout"]:
            loss["dropout_comps"] = model.dropout_comp

        loss = {**loss, **results}
        print(results)

    """
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
    """
    return loss

if __name__ == "__main__":

    # Testing only baseline in all channels!!

    channels = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
    names = ["MRI vol", "MRI cort", "Cog", "Demog", 'APOE']
    ch_type = ["long", "long", "long", "bl", 'bl']

    params = {
        "h_size": 300,
        "z_dim": 30,
        "x_hidden": 300,
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
        "n_channels": len(channels),
        "ch_names" : names,
        "ch_type": ch_type,
        "phi_layers": True,
        "sig_mean": False,
        "dropout": False,
        "drop_th": 0.3
    }

    out_dir = "experiments_mc_newloss/testing_moreparams/"
    csv_path = "data/multimodal_no_petfluid_train.csv"
    loss = run_experiment(params, csv_path, out_dir, channels)
