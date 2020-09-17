"""
File containing the model of the network.

This network is divided in three parts:

Encoder, which from an input x_t and a previous hidden state h_t-1
generates a mu and sigma that generate a latent space z_t.

Decoder, which from a sampled z_t and a hidden state h_t-1 generates
x_hat_t

RNN, that samples from z_t and with x_t and h_t-1 gives a new hidden
state h_t, for the next input.

Those three subnetworks only apply to each of the 

The three parts of the network are combined in the class model_RNNVAE

Alternative implmeentation:
https://github.com/crazysal/VariationalRNN/blob/master/VariationalRecurrentNeuralNetwork-master/model.py

"""
import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence
import os

## Decidint on device on device.
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)

#Constants
pi = torch.FloatTensor([np.pi]).to(DEVICE)  # torch.Size([1])
log_2pi = torch.log(2 * pi)

# Functions
def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
    return (logvar - torch.log(mu.pow(2) + 1e-8)).clamp(min=-8, max=8)


def compute_logvar(mu, log_alpha):
    return log_alpha + torch.log(mu.pow(2) + 1e-8)

#We will instaurate the loss as in the pytorch_modulesfrom torch.distributions import Normal, kl_divergence

def loss_has_diverged(x):
    return x[-1] > x[0]


def loss_is_nan(x):
    return str(x[-1]) == 'nan'


def moving_average(x, n=1):
    return [np.mean(x[i - n:i]) for i in range(n, len(x))]


class model_RNNVAE(nn.Module):
    """
    This class implements the full model of the network.
    We take as example the models of mcvae, to create it similar to them.
    
    We use padded inputs.

    :param x_size: size of the input
    :param h_size: size of the RNN hidden state
    :param phi_x_hidden: size of the hidden layer of the network phi_x
    :param phi_x_n: number of stacked linear layers for phi_x
    :param phi_z_hidden: size of the hidden layer of the network phi_z
    :param phi_z_n: number of stacked linear layers for phi_z
    :param enc_hidden: number of hidden layers for the encoder
    :param enc_n: number of stacked layers for the encoder
    :param latent: latent space size
    :param dec_hidden: number of hidden layers for the encoder
    :param dec_n: number of stacked layers for the encoder
    """

    def __init__(self, x_size, h_size, phi_x_hidden, phi_x_n, 
                 phi_z_hidden, phi_z_n, enc_hidden,
                 enc_n, latent, dec_hidden, dec_n, cuda=True):

        super(nn.Module, self).__init__()

        self.dtype = torch.FloatTensor
        self.use_cuda = cuda

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False

        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor

        #Parameters
        self.x_size = x_size
        self.h_size = h_size
        self.phi_x_hidden = phi_x_hidden
        self.phi_x_n = phi_x_n
        self.phi_z_hidden = phi_z_hidden
        self.phi_z_n = phi_z_n
        self.enc_hidden = enc_hidden
        self.enc_n = enc_n
        self.latent = latent
        self.dec_hidden = enc_hidden
        self.dec_n = enc_n

        # Building blocks
        ### ENCODER
        self.phi_x = self.phi_block(self.x_size, self.phi_x_hidden, self.phi_x_n)
        self.encoder, self.enc_mu, self.enc_logvar = self.var_block(self.x_size + self.h_size,
                                                                    self.enc_hidden,
                                                                    self.enc_latent,
                                                                    self.enc_n)

        ### DECODER
        self.phi_z = self.phi_block(self.latent, self.phi_z_hidden, self.phi_z_n)
        self.decoder, self.dec_mu, self.dec_logvar = self.var_block(self.latent + self.h_size,
                                                                    self.dec_hidden,
                                                                    self.x_size,
                                                                    self.dec_n)


        ### RNN 
        # TODO, ADD MORE OPTIONS, ATM ONLY GRU
        #Size is the concatenation of x + z
        #TODO: add more layers?
        self.RNN = self.nn.GRU(self.x_size + self.latent, self.h_size)

        ###INITIALIZE h0 state
        # TODO: HOW TO INITIALIZE H0 STATE?
        self.h0 = 0

        #Init KL and loss
        self.init_loss()
		self.init_KL()


    def phi_block(self, input_size, hidden_size, n_layers):
        """
        Network that transforms the features
        Uses several linear layers
        TODO: extend to add non-linearities, etc
        TODO: is there an identity layer that i could use here when phi_x_n=0?
        """
        phi_list = [] 
        for _ in range(n_layers):
            # Here we coould add non-linearities if needed
            phi_list.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        phi_model = torch.nn.ModuleList(phi_list)
        return phi_model


    def var_block(self, input_size, hidden_size, latent_size, n_layers):
        """
        Variational block that, for given parameters,
        creates a block of linear layers with a double output 
        corresponding to a mu and a logvar
        """
        var_list = []
        for _ in range(n_layers):
            # Here we coould add non-linearities if needed
            var_list.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        var_block = torch.nn.ModuleList(var_list)

        #TODO: MU COULD BE USING A SIGMOID FOR THE OUTPUT
        #TODO: LOGVAR COULD BE USING A SOFTPLUS
        to_mu = nn.Linear(input_size, latent_size)
        to_logvar = nn.Linear(input_size, latent_size)
        #Last layers, to obtain from the same tal both outputs  
        return var_block, to_mu, to_logvar

    def sample_from(self, qzx):
        """
        sampling by leveraging on the reparametrization trick
        """

        if self.training:
                zx = qzx.rsample()
        else:
            zx = qzx.loc
        return zx

    def step(self, xt, ht):
        """
        Function that implements the forward pass 
        of a single recurrent step
        TODO: what needs to be saved here? the z? the what?
        TODO: DIVIDE SO THAT Z CAN BE SAMPLED FROM
        """
        ##### ENCODER
        x_phi = self.phi_x(xt)
        # Append both inputs
        #TODO: CHECK IF THIS DIMENSION WORKS WELL
        x = torch.cat((x_phi, ht), 1)
        # Run through the encoder
        x = self.encoder(x)
        qzx_t = Normal(
            loc = self.enc_mu(x),
            scale = self.enc_logvar(x).exp.pow(0.5)
        )
        
        #Sample from
        zx_t = self.sample_from(qzx_t)

        ### DECODER
        x = torch.cat((zx_t, ht), 1)
        x = self.decoder(x)
        pxz_t = Normal(
            loc = self.dec_mu(x),
            scale = self.dec_logvar(x).exp.pow(0.5)
        )

        #Sample from
        xnext = self.sample_from(pxz_t)

        ### RNN
        x = torch.cat((zx_t, xt), 1)
        hnext, _ = self.RNN(x, ht)
        
        return xnext, hnext, zx_t, qzx_t, pxz_t


    def forward(self, x):
        """
        Forward propagation of the network,
        passing over the full network for every 
        step of the sequence
        TODO: DEBUG VERY WELL FOR DIMENSIONALITIES
        OF THE VARIOUS SEQUENCES
        """
        ht = self.Variable(torch.zeros(self.n_layers, x.size(1), self.h_size))
        # TODO. Do the xpred in a tensor way
        x_pred = []
        qzx = []
        zx = []
        pxz = []
        # ITERATE OVER THE SEQUENCE
        # and save intermediate results
        for x_t in x:
            xnext, hnext, zx_t, qzx_t, pxz_t = self.step(x_t, ht)
            x_pred.append(xnext)
            qzx.append(qzx_t)
            zx.append(zx_t)
            pxz.append(pxz_t)
            ht = hnext

        #Return similar to the mcvae implementation
        return {
            'x' : x_pred,
            'qzx': qzx,
            'zx': zx,
            'pxz': pxz
        }
    
    def reconstruct(self):
        """
        Function that will reconstruct a set of sequences
        TODO
        """
        return 0

    def sequence_predict(self, x):
        """
        Function that predicts, for a given incomplete sequence,
        future values of that sequence
        TODO VERY HARD 
        """
        return 0

    def fit_batch(self, x_batch):
        """
        Function to optimize a batch of sequences.
        """
		pred = self.forward(local_batch)
		loss = self.loss_function(pred)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.detach().item()

    def fit(self, epochs, data):
		"""
		Optimize full training.
		
        Copied from the mcvae modules
        """
		self.train()  # Inherited method which sets self.training = True

		try:  # list of epochs to allow more training
			so_far = len(self.loss['total'])
			self.epochs[-1] = so_far
			self.epochs.append(so_far + epochs)
		except AttributeError:
			self.epochs = [0, epochs]

		for epoch in range(self.epochs[-2], self.epochs[-1]):
			if type(data) is torch.utils.data.DataLoader:
				#If we have a data loader
                ##WE WILL HAVE A DATA LOADER,
                # MAKE SURE THAT THE FORMAT IS CORRECT
				current_batch = 0
				for local_batch in data:
					print("Batch # {} / {}".format(current_batch, len(data) - 1), end='\t')
					loss = self.fit_batch(local_batch)
					current_batch += 1
			else:
				loss = self.fit_batch(data)

			if np.isnan(loss):
				print('Loss is nan!')
				break

			if epoch % 100 == 0:
				self.print_loss(epoch)
				if loss_has_diverged(self.loss['total']):
					print('Loss diverged!')
					break

		self.eval()  # Inherited method which sets self.training = False

	def init_KL(self):
		# KL divergence from Pytorch
		self.KL_fn = kl_divergence

	def loss_function(self, fwd_return):
		"""
		Full loss function, as described in the paper.
        Loss function uses loss from the various
        times and stats
        """
        x = fwd_return['x']
		qzx = fwd_return['qzx']
		pxz = fwd_return['pxz']

		kl = 0
		ll = 0
        #Wstudari-ho bé
		for t in range(x.size(0)):
            kl += self.KL_fn(qzx[i], Normal(0, 1)).sum(1).mean(0)
            ll += pxz[i].log_prob(x[i]).sum(1).mean(0)

        if self.training:
			self.save_loss(losses)
			return total
		else:
			return losses

	def init_loss(self):
		empty_loss = {
			'total': [],
			'kl': [],
			'll': []
		}
		self.loss = empty_loss

	def print_loss(self, epoch):
		print('====> Epoch: {:4d}/{} ({:.0f}%)\tLoss: {:.4f}\tLL: {:.4f}\tKL: {:.4f}\tLL/KL: {:.4f}'.format(
			epoch,
			self.epochs[-1],
			100. * (epoch) / self.epochs[-1],
			self.loss['total'][-1],
			self.loss['ll'][-1],
			self.loss['kl'][-1],
			self.loss['ll'][-1] / (1e-8 + self.loss['kl'][-1])
		), end='\n')

	def save_loss(self, losses):
		for key in self.loss.keys():
			self.loss[key].append(float(losses[key].detach().item()))
