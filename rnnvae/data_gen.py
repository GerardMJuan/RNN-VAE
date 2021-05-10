"""
Small class for data generation

Some inspiration
https://stackoverflow.com/questions/22566692/python-how-to-plot-graph-sine-wave
"""

import numpy as np
import random 
from sklearn import preprocessing



class LatentTemporalGenerator():
    """
    Generate different channels of data using a number
    of latent variables that have movements through time
    """

    def __init__(self, ntp, noise=1e-3, lat_dim=5, n_channels=2, n_feats=[10, 10], variable_tp=True, sign_list=None):
        """
        Init the parameters and define the distribution
        ch_type: either "long" or "bl", if "bl", just use base z
        ntp: maximum number of time points for the longitudinal channels
        noise: amount of noise
        lat_dim: number of true latent dimensions
        n_channels: number of channels
        n_feats: number of feats, list of each channel
        variable_tp: Boolean indicating if there is a variable number of timepoints per sample
        sign_list: either None (no direction in the latent space for time) or list with len=lat_dim, with either 1 or -1.
        """
        self.ntp = ntp
        self.noise = noise # has to be a low value
        self.lat_dim = lat_dim
        self.n_channels=n_channels
        self.n_feats = n_feats        
        self.variable_tp = variable_tp
        self.sign_list = sign_list

        # Weights to transform the latent values
        self.W = []

        # Info to preprocess the data
        self.mean_train = None
        self.std_train = None

        for _ in range(self.n_channels):
            w_ = np.random.uniform(-1, 1, (self.n_feats, self.lat_dim))
            u, _, vt = np.linalg.svd(w_, full_matrices=False)
            w = u if self.n_feats >= self.lat_dim else vt
            self.W.append(w)

    def generate_samples(self, nsamples, train=True):
        """
        Generate the number of samples
        nsamples: number of samples to generate
        """

        # Create baseline latent space
        z = np.random.randn(self.lat_dim, nsamples)

        # Create the temporal information with random directions on the space per subject
        # Create a random direction per sample
        # direction depends on self.sign_list if exist, if not, 

        #Make the direction smaller such that the movement is contained
        z_v = np.random.randn(self.lat_dim, nsamples) * 0.1 
        if self.sign_list is not None:
            z_v = np.abs(z_v) * np.array(self.sign_list)[:, np.newaxis].repeat(nsamples, axis=1)

        # Create temporal timepoints
        #timepoint will be in axis 0
        
        z_temp = [z + z_v*tp for tp in range(self.ntp)]
        z_temp = np.stack(z_temp, axis=0)

        Y = []
        for ch in range(self.n_channels):
            Y_ch = []
            for z_i in z_temp:
                Y_ch.append(self.W[ch]@z_i)
            Y_ch = np.stack(Y_ch, axis=0)

            #apply noise
            Y_ch = self.noise*np.random.normal(size=Y_ch.shape)
            
            # standarize
            if train:
                self.mean_train = np.mean(Y_ch, (0, 2))
                self.std_train = np.std(Y_ch, (0, 2))

            # we broadcast the result
            Y_ch = (Y_ch - self.mean_train[None,:,None]) / self.std_train[None,:,None]
            Y.append(Y_ch)

        # dimensionality at this point
        # (ntp, nfeat, nsamples)

        # Select the number of time points 
        # for each sample
        if self.variable_tp:
            X = [np.random.choice(np.arange(0, self.ntp), length, replace=False) for length in [random.randrange(self.ntp-5, self.ntp+1) for _ in range(nsamples)]]
            X = [np.sort(x) for x in X]

            # For each channel and sample, select only the timepoints indicated in X
            # same for z_temp
            #for this reason, the shape has to change to list of subjects of shape (ntp, feat)
            Y_out = [[] for _ in range(self.n_channels)]
            z_temp_out = []
            for i in range(nsamples):
                tp_i = X[i]
                for ch in range(len(Y)):
                    Y_out[ch].append(Y[ch][:,:,i][tp_i])
                z_temp_out.append(z_temp[:,:,i][tp_i])

        # return
        return z_temp_out, Y_out


class LatentDataGeneratorCurves():
    """
    Generate different channels of data
    using a number of latent variables
    and output longitudinal channels.

    #THere are two possible paradigms for data generator. Testing the first one:
    where the curve are applied to the z 
    """

    def __init__(self, curves, ch_type, ntp, noise, lat_dim=1, n_channels=1, n_feats=10, variable_tp=True):
        """
        Init the parameters and define the distribution
        curves: the type of curves of the channels. 
        Curves is a list of tuples. Each tuple has the following aspect:
        ("type_of_curve", **par)
        "type_of_curve" can be "cos", "sin", or "sigmoid"
        ch_type: either "long" or "bl", if "bl", just use base z
        ntp: maximum number of time points for the longitudinal channels
        noise: amount of noise
        lat_dim: number of true latent dimensions
        n_channels: number of channels
        n_feats: number of feats, list of each channel
        """
        self.curves=curves
        self.ch_type = ch_type
        self.ntp = ntp
        self.noise = noise #scalar controlling the amount of gaussian noise
        self.lat_dim = lat_dim
        self.n_channels=n_channels
        self.n_feats = n_feats        
        self.variable_tp = variable_tp

        self.curve_dict ={
            "cos": cosinus,
            "sin": sinus,
            "sigmoid": sigmoid
        }
        # Weights to transform the latent values
        self.W = []

        for _ in range(self.n_channels):
            w_ = np.random.uniform(-1, 1, (self.n_feats, self.lat_dim))
            u, _, vt = np.linalg.svd(w_, full_matrices=False)
            w = u if self.n_feats >= self.lat_dim else vt
            self.W.append(w)

    def generate_samples(self, nsamples):
        """
        Generate the number of samples
        z is ds
        """
        z = np.random.randn(self.lat_dim, nsamples)

        Y = []
        for n in range(nsamples):

            #get the time points
            X = np.sort(np.random.uniform(0, 10, (self.ntp)))   

            # Create the curves
            if self.variable_tp:
                #remove n random items from the 
                length = random.choice(range(self.ntp-5, self.ntp))
                X = np.sort(np.random.uniform(0, 10, length))    

            #creat long curves
            Y_curve = []
            for ch in range(self.n_channels):
                curve = self.curves[ch]
                #for the number of features indicated in that curve
                y = self.curve_dict[curve[0]](X, curve[1])
                y = np.asarray([y_i + self.noise*np.random.normal(size=y_i.shape) for y_i in y])
                if self.ch_type[ch] == 'bl':
                    # Select only first timepoint
                    y = [y[0]]
                Y_curve.append(preprocessing.scale(y))
            Y.append(Y_curve)
            
        self.Y_out = []
        for ch in range(self.n_channels):
            #Multiply to normal space
            Y_ch = self.W[ch]@z
            # Standarize 
            Y_ch = preprocessing.scale(Y_ch)
            # sum to curves
            Y_ch_full = []
            for n in range(nsamples):
                Y_temp = np.array([Y_ch[:,n] + cur for cur in Y[n][ch]])
                Y_ch_full.append(Y_temp)

            self.Y_out.append(Y_ch_full)
        return z, self.Y_out
