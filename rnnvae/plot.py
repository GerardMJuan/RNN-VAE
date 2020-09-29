
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


def plot_trajectory(X, X_hat, subj, feat, out_dir, out_name):
    # Plot trajectory for a given subjects or features comparing the predicted values with
    # the actual values. For a given subject or feature (index is passed) or for all the features
    #or subjects, doing a mean
    # X_hat and X are lists of (nt x nfeat) We assume that the dimensions of each element correspond  
    #TODO: check if all the dimensions are correct and agree with the actual programming
    if feat=='all':
        feat = list(range(X_hat[0].shape[1]))
    # Not really sure this works. This needs to be validated
    X_hat = X_hat[subj].numpy()   #Select only the subject we want
    X = X[subj]   #Select only the subject we want

    if feat=='all':
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

def plot_z(x, dims=[0,1]):
    # x contains the latent space codes for a given number of subjects
    # dims are the two dimensions that will be ploted
    print('NYI')


def plot_z_time_1d():
    #Plot a latent dimension like a 2d graph, over a set of time points and 
    print('NYI')

def plot_z_time_2d():

    print('NYI')