import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_losses(loss_curve, out_dir, out_name):
    # Plot the loss curve
    #loss_curve is a dict with keys kl, ll and total
    # (as in the model)
    plt.figure()

    plt.plot(range(len(loss_curve['kl'])), loss_curve['kl'], '-', label='KL loss')
    plt.plot(range(len(loss_curve['ll'])), loss_curve['ll'], '-', label='LL loss')
    plt.plot(range(len(loss_curve['total'])), loss_curve['total'], '-', label='total loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(loc='upper left')
    plt.title("Losses")

    plt.savefig(out_dir + out_name + '.png')
    plt.close()

def plot_total_loss(train_loss, val_loss, name, out_dir, out_name):
    #Compare the total loss of train and validation.
    # train loss and val loss are directly the actual values
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, '-', label='Train loss')
    plt.plot(range(len(val_loss)), val_loss, '-', label='Validation loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(loc='upper left')
    plt.title(name)

    plt.savefig(out_dir + out_name + '.png')
    plt.close()

def plot_many_trajectories(X, feat, ntp, out_dir, out_name):
    """
    Simple function that plots all subjects, for a single features (or all) 
    over "time".
    Should extend to later things (age, etc)
    TODO: adapt it to variable number of timepoints
    """
    if feat=='all':
        feat = list(range(X.shape[2]))
        X = np.mean(X[:, :, feat], axis=2)
    else:
        X = X[:, feat]

    sns.set_theme()
    sns.set_context("paper")

    # create figure, plot it
    plt.figure(figsize=(10,10))
    for line in X:
        plt.plot(list(range(len(line))), line, linewidth=0.5)

    mean_x_axis = list(range(ntp))
    ys_interp = [np.interp(mean_x_axis, list(range(len(x))), x) for x in X]
    mean_y_axis = np.mean(ys_interp, axis=0)

    #plot it too
    #This needs to be plotted differently from the others
    plt.plot(mean_x_axis, mean_y_axis, linewidth=3)
    
    plt.xlabel(f"Timepoints", size=13)
    plt.ylabel(f"Y", size=13)

    plt.savefig(out_dir + out_name + '.png')
    plt.close()




def plot_trajectory(X, X_hat, subj, feat, out_dir, out_name):
    # Plot trajectory for a given subjects or features comparing the predicted values with
    # the actual values. For a given subject or feature (index is passed) or for all the features
    #or subjects, doing a mean
    # X_hat and X are lists of (nt x nfeat) We assume that the dimensions of each element correspond  
    #TODO: check if all the dimensions are correct and agree with the actual programming

    # Not really sure this works. This needs to be validated
    X_hat = X_hat[subj]   #Select only the subject we want
    X = X[subj]   #Select only the subject we want

    if feat=='all':
        feat = list(range(X_hat.shape[1]))
        x_hat_curve = np.mean(X_hat[:, feat], axis=1)
        x_val_curve = np.mean(X[:, feat], axis=1)
    else:
        x_hat_curve = X_hat[:, feat]
        x_val_curve = X[:, feat]    

    # Plot the two lines
    plt.plot(range(len(x_hat_curve)), x_hat_curve, '-b', label='X (predicted)')
    plt.plot(range(len(x_val_curve)), x_val_curve, '-r', label='X (original)')

    plt.xlabel("time-point")
    plt.ylabel("value")

    plt.legend(loc='upper left')
    plt.title("Predicted vs real")

    plt.savefig(out_dir + out_name + '.png')
    plt.close()

def plot_z_2d(x0, x1, z_t, color, dims, out_dir, out_name='latent_space'):
    # x0 and x1 contains the latent space codes for a given number of subjects
    # dims are the two dimensions that will be ploted
    #color is a vector same length as x[0] indicating color of space
    sns.set_theme()
    sns.set_context("paper")

    dim0 = dims[0]
    dim1 = dims[1]

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=x0, y=x1, hue=color, s=60)
    plt.xlabel(f"Dim {dim0}", size=13)
    plt.ylabel(f"Dim {dim1}", size=13)
    plt.title(f"Latent space for z_{z_t}", size=15)
    plt.savefig(f'{out_dir}/{out_name}.png')
    plt.close()

def plot_z_time_1d():
    #Plot a latent dimension like a 2d graph, over a set of time points and 
    #LOok at the 
    print('NYI')


def plot_z_time_2d(z, max_timepoints, dims, out_dir, c='tp', Y=None, out_name='latent_space_2d'):
    """
    Plot two dimension of the latent space with all the timepoints there,
    c parameter can be time point, or any other value in the Y dictionary
    """
    sns.set_theme()
    sns.set_context("paper")
    plt.figure(figsize=(10, 10))

    # create color cmap
    if c == 'DX':
        pallete = sns.color_palette(["#2a9e1e", "#bfbc1a", "#af1f1f"])
        #apply to Y
        dx_dict = {
            "NL": "CN",
            "MCI": "MCI",
            "MCI to NL": "CN",
            "Dementia": "AD",
            "Dementia to MCI": "MCI",
            "NL to MCI": "MCI",
            "NL to Dementia": "AD",
            "MCI to Dementia": "AD"
        }
    else:
        pallete = sns.color_palette("viridis", as_cmap=True)

    z_d0_full = []
    z_d1_full = []
    color = []
    dim0 = dims[0]
    dim1 = dims[1]
    for tp in range(max_timepoints):

        z_d0 = [x[tp, dim0] for x in z if x.shape[0] > tp]
        z_d1 = [x[tp, dim1] for x in z if x.shape[0] > tp]
        
        #populate 
        z_d0_full = z_d0_full + z_d0
        z_d1_full = z_d1_full + z_d1

        # colorise
        if c == 'tp':
            color = color + [tp]*(len(z_d0))
        elif c == 'DX':
            color = color + [dx_dict[x[tp]] for x in Y[c] if x.shape[0] > tp]
        else:
            color = color + [x[tp] for x in Y[c] if x.shape[0] > tp]

    print(len(z_d0_full))
    print(len(z_d1_full))
    print(len(color))
    sns.scatterplot(x=z_d0_full,y=z_d1_full,hue=color, palette=pallete, s=60)
    ##Add title, x and y axis
    plt.xlabel(f"Dim {dim0}", size=13)
    plt.ylabel(f"Dim {dim1}", size=13)
    plt.title(f"Latent space all timepoints, colored by {c}", size=15)
    plt.savefig(f'{out_dir}/{out_name}.png')
    plt.close()