"""
Script with all the new parameters added, used for testing those parameters.

Using cross-validation
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from rnnvae import rnnvae_h
from rnnvae.utils import load_multimodal_data_cv
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

    # Begin on the CV data
    gen = load_multimodal_data_cv(csv_path, data_cols, p["ch_type"], nsplit=10, normalize=True)
    # Prepare the data structures for the data

    # Load the different folds
    loss = {
        "mae_train" : [],
        "rec_train" : [],
        "mae_test": [],
        "loss_total": [],
        "loss_total_val": [],
        "loss_kl": [],
        "loss_ll": [],
    }

    pred_results = {}
    for ch_name in p["ch_names"][:3]:
        pred_results[f"pred_{ch_name}_mae"] = []

    rec_results = {}
    for ch_name in p["ch_names"]:
        rec_results[f"recon_{ch_name}_mae"] = []

    loss = {**loss, **pred_results, **rec_results}

    # iterator marking the fold
    fold_n = 0

    for X_train, X_test, Y_train, Y_test, mri_col in gen:
    # LOAD DATA
    #Start by not using validation data
    # this is a list of values

        #Create output dir for the fold
        out_dir_cv = out_dir + f'_fold_{fold_n}/'
        if not os.path.exists(out_dir_cv):
            os.makedirs(out_dir_cv)

        #Redirect output to specific folder
        sys.stdout = open(out_dir_cv + 'output.out', 'w')
        
        p["n_feats"] = [x[0].shape[1] for x in X_train]

        X_train_list = []
        mask_train_list = []

        X_test_list = []
        mask_test_list = []

        print('Length of train/test')
        print(len(X_train[0]))
        print(len(X_test[0]))

        # need to deal with ntp here
        ntp = max(np.max([[len(xi) for xi in x] for x in X_train]), np.max([[len(xi) for xi in x] for x in X_train]))

        if p["long_to_bl"]:
            # HERE, change bl to long and repeat the values at t0 for ntp
            for i in range(len(p["ch_type"])):
                if p["ch_type"][i] == 'bl':
                    for j in range(len(X_train[i])):
                        X_train[i][j] = np.array([X_train[i][j][0]]*ntp) 

                    for j in range(len(X_test[i])):
                        X_test[i][j] = np.array([X_test[i][j][0]]*ntp) 

                    # p["ch_type"][i] = 'long'
        
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

        #ntp = max(max([x.shape[0] for x in X_train_list]), max([x.shape[0] for x in X_train_list]))

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
        model.fit(X_train_list, X_test_list, mask_train_list, mask_test_list)

        #fit the model after changing the lr
        #optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"]*.1)
        #model.optimizer = optimizer
        #print('Refining optimization...')
        #model.fit(X_train_list, X_test_list, mask_train_list, mask_test_list)

        if p["dropout"]:
            print("Print the dropout")
            print(model.dropout_comp)

        ### After training, save the model!
        model.save(out_dir_cv, 'model.pt')

        # Predict the reconstructions from X_val and X_train
        X_train_fwd = model.predict(X_train_list, mask_train_list, nt=ntp)
        X_test_fwd = model.predict(X_test_list, mask_test_list, nt=ntp)

        # Unpad using the masks
        #plot validation and 
        plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir_cv, "total_loss.png")
        plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir_cv, "kl_loss.png")
        plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir_cv, "ll_loss.png") #Negative to see downard curve

        #Compute mse and reconstruction loss
        #General mse and reconstruction over 
        # test_loss = model.recon_loss(X_test_fwd, target=X_test_pad, mask=mask_test_tensor)
        train_loss = model.recon_loss(X_train_fwd, target=X_train_list, mask=mask_train_list)
        test_loss = model.recon_loss(X_test_fwd, target=X_test_list, mask=mask_test_list)

        print('MSE over the train set: ' + str(train_loss["mae"]))
        print('Reconstruction loss over the train set: ' + str(train_loss["rec_loss"]))

        print('MSE over the test set: ' + str(test_loss["mae"]))
        print('Reconstruction loss the train set: ' + str(test_loss["rec_loss"]))
        
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
                loss[f"recon_{curr_name}_mae"].append(mae_rec)

        # Save results in the loss object
        loss["mae_train"].append(train_loss["mae"])
        loss["rec_train"].append(train_loss["rec_loss"])
        loss["mae_test"].append(train_loss["mae"])
        loss["loss_total"].append(model.loss['total'][-1])
        loss["loss_total_val"].append(model.val_loss['total'][-1])
        loss["loss_kl"].append(model.loss['kl'][-1])
        loss["loss_ll"].append(model.loss['ll'][-1])

        fold_n += 1
        # break at 5 iterations, need to do it faster
        if fold_n == 2:
            break

    # Compute the mean for every param in the loss dict
    for k in loss.keys():
        loss[k] = np.mean(loss[k])
    
    print(loss)
    return loss


if __name__ == "__main__":

    ### Parameter definition

    #channels = ['_mri_vol','_mri_cort','_demog','_apoe', '_cog', '_fluid','_fdg','_av45']
    #names = ["MRI vol", "MRI cort", "Demog", "APOE", "Cog", "Fluid", "FDG", "AV45"]

    channels = ['_mri_vol','_mri_cort', '_cog', '_demog', '_apoe']
    names = ["MRI vol", "MRI cort", "Cog", "Demog", 'APOE']
    ch_type = ["long", "long", "long", "bl", 'bl']
    constrain = [None, None, 5, 5, 5]

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
        "n_epochs": 2000,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
        "n_channels": len(channels),
        "ch_names" : names,
        "ch_type": ch_type,
        "c_z": constrain,
        "phi_layers": True,
        "sig_mean": False,
        "dropout": True,
        "drop_th": 0.4,
        "long_to_bl": True
    }

    out_dir = "/homedtic/gmarti/EXPERIMENTS/RNNVAE/constrained/tests/"
    csv_path = "data/multimodal_no_petfluid_train.csv"
    loss = run_experiment(params, csv_path, out_dir, channels)
