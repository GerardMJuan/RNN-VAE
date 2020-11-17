"""
test_mc_synth.py

Testing for a single channel for synthetic,
longitudinal data. This file will be used. We will generate
different length signals to input to our model and see how it behaves.

Also, two different settings:
- generate them at different length but starting from the same time point
- generate them at different length and also at different time points.
"""

#imports
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import torch
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
    X_train_tensor = []
    X_test_tensor = []
    #generate the data
    for ch_curves in p['curves']:
        gen_model = SinDataGenerator(ch_curves, p["ntp"], p["noise"])
        samples = gen_model.generate_n_samples(p["nsamples"])
        X_train = np.asarray([y for (_,y) in samples])
        X_train_tensor.append(torch.FloatTensor(X_train).permute((1,0,2)).to(DEVICE))

        samples = gen_model.generate_n_samples(int(p["nsamples"]*0.8))
        X_test = np.asarray([y for (_,y) in samples])
        X_test_tensor.append(torch.FloatTensor(X_test).permute((1,0,2)).to(DEVICE))
    
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
                            p["n_channels"], p["n_feats"], p["model_name_dict"], DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    model.optimizer = optimizer

    model = model.to(DEVICE)
    # Fit the model
    model.fit(X_train_tensor, X_test_tensor)

    ### After training, save the model!
    model.save(out_dir, 'model.pt')

    # Predict the reconstructions from X_val and X_train
    X_test_fwd = model.predict(X_test_tensor, nt=p["ntp"])
    X_train_fwd = model.predict(X_train_tensor, nt=p["ntp"])

    # Unpad using the masks
    #plot validation and 
    plot_total_loss(model.loss['total'], model.val_loss['total'], "Total loss", out_dir, "total_loss.png")
    plot_total_loss(model.loss['kl'], model.val_loss['kl'], "kl_loss", out_dir, "kl_loss.png")
    plot_total_loss(model.loss['ll'], model.val_loss['ll'], "ll_loss", out_dir, "ll_loss.png") #Negative to see downard curve


    #Compute mse and reconstruction loss
    test_loss = model.recon_loss(X_test_fwd, target=X_test_fwd['xnext'])
    train_loss = model.recon_loss(X_train_fwd, target=X_train_fwd['xnext'])

    print('MSE over the train set: ' + str(train_loss["mae"]))
    print('Reconstruction loss over the train set: ' + str(train_loss["rec_loss"]))

    print('MSE over the test set: ' + str(test_loss["mae"]))
    print('Reconstruction loss the train set: ' + str(test_loss["rec_loss"]))

    #Sampling
    # Create first samples with only one timepoint
    X_samples_tensor = []
    nsamples = 500
    #generate the data
    for ch_curves in p['curves']:

        gen_model = SinDataGenerator(ch_curves, p["ntp"], p["noise"])
        samples = gen_model.generate_n_samples(nsamples)
        X_samples = np.asarray([y[:1] for (_,y) in samples])
        X_samples_tensor.append(torch.FloatTensor(X_samples).permute((1,0,2)).to(DEVICE))

    X_sample = model.predict(X_samples_tensor, p['ntp'])

    #Get the samples
    X_pred = [np.array(x).swapaxes(0,1) for x in X_sample['xnext']]
    z_sample = [np.array(x).swapaxes(0,1) for x in X_sample['z']]

    # plot the samples over time
    for x_ch, ch_name in zip(X_pred, p["model_name_dict"].values()):
        plot_many_trajectories(x_ch, 'all', p["ntp"], out_dir, f'ch_{ch_name}_x_samples')

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

    #plot latent space
    for ch in range(p["n_channels"]):
        for dim0 in range(p["z_dim"]):
            for dim1 in range(dim0, p["z_dim"]):
                if dim0 == dim1: continue   # very dirty
                plot_z_time_2d(z_sample[ch], p["ntp"], [dim0, dim1], out_dir + sampling_path, out_name=f'z_ch_{ch}_d{dim0}_d{dim1}')


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

    plot_latent_space(model, qzx, p["ntp"], classificator=classif, plt_tp='all',
                    all_plots=False, uncertainty=True, savefig=True, out_dir=out_dir_sample)

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
        [("sigmoid", {"L": 1, "k": 1, "x0": 5}),    
        ("sin", {"A": 1, "f": 0.2}),
        ("cos", {"A": 1, "f": 0.2})],
        [("sigmoid", {"L": 1, "k": -15, "x0": 5}),
        ("sigmoid", {"L": 1, "k": 5, "x0": 5})]
        ]
    
    names = {"0":"c1", 
             "1":"c2"}

    ### Parameter definition
    params = {
        "h_size": 20,
        "z_dim": 5,
        "hidden": 20,
        "n_layers": 1,
        "n_epochs": 1500,
        "clip": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "seed": 1714,
        "curves": curves,
        "ntp": 15,
        "noise": 0.2,
        "nsamples": 300,
        "n_channels": len(curves),
        "n_feats": [len(x) for x in curves],
        "model_name_dict": names
    }

    out_dir = "experiments_mc/synth_nopadding/"
    loss = run_experiment(params, out_dir)