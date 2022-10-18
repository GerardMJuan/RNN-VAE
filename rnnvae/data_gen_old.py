
import numpy as np
import random 
from sklearn import preprocessing



def sinus(x, p):
    """
    Compute sinusoid function.
    pars is a dict containing the parameters
    """
    y = p["A"]*np.sin(2*np.pi*p["f"]*x)
    return y

def cosinus(x, p):
    """
    Compute cosinus function.
    pars is a dict containing the parameters
    """
    y = p["A"]*np.cos(2*np.pi*p["f"]*x)
    return y

def sigmoid(x, p):
    """
    Compute sigmoid function, using logistic formula
    pars is a dict containing the parameters
    """
    y = p["L"]/(1 + np.exp(-p["k"]*(x-p["x0"])))
    return y 



class LatentDataGenerator():
    """
    Generate different channels of data
    using a number of latent variables
    and output longitudinal channels.

    #THere are two possible paradigms for data generator. Testing the first one:
    where the curve are applied to the z 
    """

    def __init__(self, ch_type, ntp, noise, lat_dim=1, n_channels=1, n_feats=10):
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
        self.ch_type = ch_type
        self.ntp = ntp
        self.noise = noise #scalar controlling the amount of gaussian noise
        self.lat_dim = lat_dim
        self.n_channels=n_channels
        self.n_feats = n_feats        

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
        z = np.random.randn(nsamples, self.lat_dim)

        #get the time points
        X = np.sort(np.random.uniform(0, 1, (self.lat_dim, self.ntp)))   

        #Create the latent timeshits
        self.Z_t = []

        for i in range(self.ntp):
            y = z*X[:,i]

            # y = self.curve_dict[curve[0]](X, curve[1])
            # y = np.asarray([z for z in y])
            self.Z_t.append(y)

        self.Y = []
        # Create the outputs
        for ch in range(self.n_channels):
            #Multiply to each z_t
            Y_ch = np.array([self.W[ch]@Z.T for Z in self.Z_t])
            Y_ch = Y_ch + self.noise*np.random.normal(size=Y_ch.shape)
            if self.ch_type[ch] == 'bl':
                # Select only first timepoint
                Y_ch = Y_ch[0,:]
            self.Y.append(Y_ch)
        return self.Z_t, self.Y


class SinDataGenerator():

    def __init__(self, curves, ntp, noise, variable_tp=False):
        """
        Init the parameters and define the distribution

        Curves is a list of tuples. Each tuple has the following aspect:
        ("type_of_curve", **par)
        "type_of_curve" can be "cos", "sin", or "sigmoid"
        par is a dictionary with the parameters used by each signal
        Parameteres of 
        """
        self.curves = curves
        self.ntp = ntp
        self.noise = noise #scalar controlling the amount of gaussian noise
        self.variable_tp = variable_tp #boolean controlling if the number of timepoints is variable or not

        self.curve_dict ={
            "cos": self.cosinus,
            "sin": self.sinus,
            "sigmoid": self.sigmoid
        }

    def generate_full_signal(self):
        """
        Using the base linspace, generate full signal f(x) without noise. 
        """
        #Use a linspace from zero to 10
        x = np.linspace(0, 10, 5000)
        y_total = []

        for curve in self.curves:
            #for the number of features indicated in that curve
            y = self.curve_dict[curve[0]](x, curve[1])
            y_total.append(y)

        return x, np.array(y_total).T


    def sinus(self, x, p):
        """
        COmpute sinusoid function.
        pars is a dict containing the parameters
        """
        y = p["A"]*np.sin(2*np.pi*p["f"]*x)
        return y

    def cosinus(self, x, p):
        """
        COmpute cosinus function.
        pars is a dict containing the parameters
        """
        y = p["A"]*np.cos(2*np.pi*p["f"]*x)
        return y


    def sigmoid(self, x, p):
        """
        Compute sigmoid function, using logistic formula
        pars is a dict containing the parameters
        """
        y = p["L"]/(1 + np.exp(-p["k"]*(x-p["x0"])))
        return y - (p["L"]/2)

    def sample(self):
        """
        Generate a single sample with the parameters from the generator
        NYI: NON EQUAL SPACING, VARIABLE TPS
        """
        #X = np.linspace(0, 10, self.ntp)
        X = np.sort(np.random.uniform(0, 10, self.ntp))
        #X = np.zeros(self.ntp)
        if self.variable_tp:
            #remove n random items from the 
            length = random.choice(range(self.ntp-5, self.ntp))
            X = np.sort(np.random.uniform(0, 10, length))    
            # If we have variable tp, we remove random points in this            
        # X = np.asarray([x + self.noise*np.random.normal(size=x.shape) for x in X])
        Y = []
        for curve in self.curves:
            #for the number of features indicated in that curve
            y = self.curve_dict[curve[0]](X, curve[1])
            y = np.asarray([z + self.noise*np.random.normal(size=z.shape) for z in y])
            Y.append(y)
        return np.asarray(X), np.asarray(Y).T

    def generate_n_samples(self, nsamples):
        
        """
        generate nsamples from the distribution.

        Returns a list of nsamples with elements of shape (nt, )
        """
        samples = []
        for n in range(nsamples):
            samples.append(self.sample())
        return samples
            
