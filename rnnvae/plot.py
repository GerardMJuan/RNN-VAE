import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from matplotlib.patches import Ellipse

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


def plot_z_time_2d(z, max_timepoints,dims, out_dir, out_name='latent_space_2d'):
    """
    Plot two dimension of the latent space with all the timepoints there,
    colored by timepoints
    """
    sns.set_theme()
    sns.set_context("paper")
    plt.figure(figsize=(10, 10))

    # create color cmap
    pallete = sns.color_palette("viridis", max_timepoints)

    z_d0_full = []
    z_d1_full = []
    tp_full = []
    dim0 = dims[0]
    dim1 = dims[1]
    for tp in range(max_timepoints):

        z_d0 = [x[tp, dim0] for x in z if x.shape[0] > tp]

        z_d1 = [x[tp, dim1] for x in z if x.shape[0] > tp]
        
        #populate 
        z_d0_full = z_d0_full + z_d0
        z_d1_full = z_d1_full + z_d1
        tp_full = tp_full + [tp]*(len(z_d0))


    sns.scatterplot(x=z_d0_full,y=z_d1_full,hue=tp_full, palette="viridis", s=60)
    ##Add title, x and y axis
    plt.xlabel(f"Dim {dim0}", size=13)
    plt.ylabel(f"Dim {dim1}", size=13)
    plt.title(f"Latent space all timepoints, colord by tp", size=15)
    plt.savefig(f'{out_dir}/{out_name}.png')
    plt.close()


def plot_latent_space(model, qzx, max_tp, classificator=None, plt_tp='all', text=None, all_plots=False, uncertainty=True, comp=None, savefig=False, out_dir=None, mask=None):
    """
    Copied from MCVAE.

    qzx is already processed data from the decoder.
    qzx should be of the form [nch, ntp] and converted to cpu
    Plot the latent space on the data.
    Adapted for temporal data.
    This adaptation means that we add a new parameter named time,
    Which takes into account which timepoint to plot (can be 'all')
    # If we want to classify by time, use the classificator
    parameter with the timepoint info OUTSIDE the tal.

    #Classificator should correspond to the length of the data and tp indicated.

    The mask indicates the subjects that are present for that timepoint and channel, and as such, the ones that should be plotted.
    """
    sns.reset_defaults()    
    channels = len(qzx)
    comps = model.latent
    if mask is not None: mask_ch = [mask_ch[t, :, 0] for mask_ch in mask]

    if classificator is not None:
        groups = np.unique(classificator)

    # One figure per latent component
    #  Linear relationships expected between channels
    if comp is not None:
        itercomps = comp if isinstance(comp, list) else [comp]
    else:
        itercomps = range(comps)
    # For each component
    for comp in itercomps:
        fig, axs = plt.subplots(channels, channels, figsize=(20,20))
        fig.suptitle(r'$z_{' + str(comp) + '}$', fontsize=30)
        for i, j in itertools.product(range(channels), range(channels)):
            ax = axs if channels == 1 else axs[j, i]
            if i == j:
                ax.text(
                    0.5, 0.5, 'z|{}'.format(model.ch_name[i]),
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=20
                )
                ax.axis('off')
            elif i > j:
                xi, xj, si, sj = (np.array([]) for i in range(4))

                #Select the mask
                if mask is not None: mask_ij = [mask_ch[t, :, 0] for mask_ch in mask]

                # select only the timepoints needed
                # We assume that all subjects have the same number of timepoints
                for tp in range(max_tp):
                    if plt_tp != "all" and tp not in plt_tp:
                        continue
                    xii = qzx[i][tp].loc.cpu().detach().numpy()[:, comp]
                    xjj = qzx[j][tp].loc.cpu().detach().numpy()[:, comp]
                    sii = qzx[i][tp].scale.cpu().detach().numpy()[:, comp]
                    sjj = qzx[j][tp].scale.cpu().detach().numpy()[:, comp]
                    #If we have mask, remove the points that, for that two channels, apply
                    import pdb; pdb.set_trace()
                    if mask is not None:
                        xii = xii[mask_ij]
                        xjj = xjj[mask_ij]
                        sii = sii[mask_ij]
                        sjj = sjj[mask_ij]

                    #Go subject by subject and append the information
                    xi = np.append(xi, xii)
                    xj = np.append(xj, xj)
                    si = np.append(si, si)
                    sj = np.append(sj, sj)
                ells = [Ellipse(xy=[xi[p], xj[p]], width=2 * si[p], height=2 * sj[p]) for p in range(len(xi))]
                if classificator is not None:
                    #For this to work, length of classificator must be equal to length of the timepoints and subjects 
                    for g in groups:
                        g_idx = classificator == g
                        ax.plot(xi[g_idx], xj[g_idx], '.', alpha=0.75, markersize=15)
                        if uncertainty:
                            color = ax.get_lines()[-1].get_color()
                            for idx in np.where(g_idx)[0]:
                                ax.add_artist(ells[idx])
                                ells[idx].set_alpha(0.1)
                                ells[idx].set_facecolor(color)
                else:
                    ax.plot(xi, xj, '.')
                    if uncertainty:
                        for e in ells:
                            ax.add_artist(e)
                            e.set_alpha(0.1)
                if text is not None:
                    [ax.text(*item) for item in zip(xi, xj, text)]
                # Bisettrice
                lox, hix = ax.get_xlim()
                loy, hiy = ax.get_ylim()
                lo, hi = np.min([lox, loy]), np.max([hix, hiy])
                ax.plot([lo, hi], [lo, hi], ls="--", c=".3")
            else:
                ax.axis('off')
        if classificator is not None:
            groups = sorted(groups, key=lambda t: int(t))
            [axs[-1, 0].plot(0,0) for g in groups]
            legend = ['{} (n={})'.format(g, len(classificator[classificator==g])) for g in groups]
            axs[-1,0].legend(legend)
            try:
                axs[-1, 0].set_title(classificator.name)
            except AttributeError:
                axs[-1, 0].set_title('Groups')

        #save figure
        if savefig:
            plt.savefig(f"{out_dir}latent_space_zcomp_{comp}.png")
            plt.close()

    if all_plots:  # comps > 1:
        # TODO: remove based on components
        # One figure per channel
        #  Uncorrelated relationsips expected between latent components
        for ch in range(channels):
            fig, axs = plt.subplots(comps, comps, figsize=(20,20))
            fig.suptitle(model.ch_name[ch], fontsize=30)
            for i, j in itertools.product(range(comps), range(comps)):
                if i == j:
                    axs[j, i].text(
                        0.5, 0.5, r'$z_{' + str(i) + '}$',
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=20
                    )
                    axs[j, i].axis('off')
                elif i > j:
                    xi, xj = (np.array([]) for i in range(2))
                    for tp in range(max_tp):
                        if plt_tp != "all" and tp not in plt_tp:
                            continue
                        xi = np.append(xi, qzx[ch][tp].loc.cpu().detach().numpy()[:, i])
                        xj = np.append(xj, qzx[ch][tp].loc.cpu().detach().numpy()[:, j])
                    if classificator is not None:
                        for g in groups:
                            g_idx = classificator == g
                            axs[j, i].plot(xi[g_idx], xj[g_idx], '.')
                    else:
                        axs[j, i].plot(xi, xj, '.')
                    # zero axis
                    axs[j, i].axhline(y=0, ls="--", c=".3")
                    axs[j, i].axvline(x=0, ls="--", c=".3")
                else:
                    axs[j, i].axis('off')
            if classificator is not None:
                groups = sorted(groups, key=lambda t: int(t))
                [axs[-1, 0].plot(0,0) for g in groups]
                legend = ['{} (n={})'.format(g, len(classificator[classificator==g])) for g in groups]
                axs[-1,0].legend(legend)

            #save figure
            if savefig:
                plt.savefig(f"{out_dir}latent_space_ch_{model.ch_name[ch]}.png")
                plt.close()