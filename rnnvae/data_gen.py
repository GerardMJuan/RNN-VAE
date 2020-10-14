"""
Small class for data generation

Some inspiration
https://stackoverflow.com/questions/22566692/python-how-to-plot-graph-sine-wave
"""

import numpy as np
import random 

class SinDataGenerator():

    def __init__(self, curves, ntp, noise, variable_tp=False, equal_spacing=True):
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
        self.equal_spacing = equal_spacing #boolean controlling if the samples over time are equally spaced or not

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
        return y 

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
            
