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


class PhiBlock(nn.Module):
    """
    This class implement the phi subblock that is used to transform inputs
    of the full network.
    """

    def __init__(self, input_size, hidden_size, n_layers):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        #Define module list
        self.init_block()
 
    def init_block(self):
        """
        Auxliar function to initalize the block
        TODO: extend to add non-linearities, etc
        TODO: is there an identity layer that i could use here when phi_x_n=0?
        """
        self.phi_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            # Here we coould add non-linearities if needed
            self.phi_layers.append(nn.Linear(self.input_size, self.hidden_size))
            self.input_size = self.hidden_size

    def forward(self, x):
        """
        Forward pass of the ModuleList
        """
        for i in range(len(self.phi_layers)):
            x = self.phi_layers[i](x)
        return x

class VariationalBlock(nn.Module):
    """ 
    Class implmenting the variational subblock for the VAE network,
    receiving an input and the forward pass generating a Normal distribution.
    """

    def __init__(self, input_size, hidden_size, latent_size, n_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.n_layers = n_layers

        #Define module list
        self.init_block()


    def init_block(self):
        self.var_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            # Here we coould add non-linearities if needed
            self.var_layers.append(nn.Linear(self.input_size, self.hidden_size))
            self.input_size = self.hidden_size

        #TODO: MU COULD BE USING A SIGMOID FOR THE OUTPUT
        #TODO: LOGVAR COULD BE USING A SOFTPLUS
        self.to_mu = nn.Linear(self.input_size, self.latent_size)
        self.to_logvar = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
            nn.Softplus())
        #Last layers, to obtain from the same tal both outputs  


    def forward(self, x):
        for i in range(len(self.var_layers)):
            x = self.var_layers[i](x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        x = Normal(
            loc = mu,
            scale = logvar.exp().pow(0.5)
        )
        return x


class ModelRNNVAE(nn.Module):
    """
    This class implements the full model of the network.
    We take as example the models of mcvae, to create it similar to them.
    
    Note: inputs to the network should already be padded, so we should consider
    that all sequences have the same length, be padded accordingly.

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
                 enc_n, latent, dec_hidden, dec_n, clip, cuda=True,
                 print_every=1):

        super().__init__()

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

        # other parameters
        self.clip = clip
        self.print_every = print_every

        # Building blocks
        ## PRIOR 
        self.prior  = VariationalBlock(self.h_size, self.enc_hidden, self.latent, self.enc_n)

        ### ENCODER
        self.phi_x = PhiBlock(self.x_size, self.phi_x_hidden, self.phi_x_n)
        self.encoder = VariationalBlock(self.phi_x_hidden + self.h_size, self.enc_hidden, self.latent, self.enc_n)

        ### DECODER
        self.phi_z = PhiBlock(self.latent, self.phi_z_hidden, self.phi_z_n)
        self.decoder = VariationalBlock(self.phi_z_hidden + self.h_size, self.dec_hidden, self.x_size, self.dec_n)

        ### RNN 
        # TODO, ADD MORE OPTIONS, ATM ONLY GRU
        #Size is the concatenation of phi_x + phi_z
        #TODO: add more layers? right now, nlayers is hardcoded to 1
        self.RNN = nn.GRU(self.phi_x_hidden + self.phi_z_hidden, self.h_size, 1)

        #Init KL and loss
        self.optimizer = None
        self.init_KL()

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
        """
        ###PRIOR
        # RETURN THE PRIOR
        z_prior = self.prior(ht[-1])

        ##### ENCODER
        x_phi = self.phi_x(xt)
        # Append both inputs
        x = torch.cat([x_phi, ht[-1]], 1)
        # Run through the encoder
        qzx_t = self.encoder(x)
        
        #Sample from
        zx_t = self.sample_from(qzx_t)

        #Apply phi_z
        phi_zx_t = self.phi_z(zx_t)

        ### DECODER
        x = torch.cat([phi_zx_t, ht[-1]], 1)
        pxz_t = self.decoder(x)

        #Sample from
        xnext = self.sample_from(pxz_t)

        ### RNN
        x = torch.cat([phi_zx_t, x_phi],1).unsqueeze(0)
        _, hnext = self.RNN(x, ht)
        
        return xnext, hnext, z_prior, zx_t, qzx_t, pxz_t


    def forward(self, x):
        """
        Forward propagation of the network,
        passing over the full network for every 
        step of the sequence
        TODO: DEBUG VERY WELL FOR DIMENSIONALITIES
        OF THE VARIOUS SEQUENCES
        """
        # Initial h0
        ht = Variable(torch.zeros(1, x.size(1), self.h_size))
        x_pred = []
        qzx = []
        zx = []
        pxz = []
        zp = []
        # ITERATE OVER THE SEQUENCE
        # and save intermediate results
        for x_t in x:
            xnext, hnext, zp_t, zx_t, qzx_t, pxz_t = self.step(x_t, ht)
            x_pred.append(xnext)
            qzx.append(qzx_t)
            zx.append(zx_t)
            pxz.append(pxz_t)
            zp.append(zp_t)
            ht = hnext

        #Return similar to the mcvae implementation
        return {
            'x' : x_pred,
            'qzx': qzx,
            'zx': zx,
            'pxz': pxz,
            'zp': zp
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
        pred = self.forward(x_batch)
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
                    #print("Batch # {} / {}".format(current_batch, len(data) - 1), end='\n')
                    #happens if the dataset has labels, discard them
                    if type(local_batch) is list:
                        local_batch = local_batch[0]
                    # Study well why this transpose and everything
                    #Will probably be removed when using real data
                    data = Variable(local_batch.squeeze().transpose(0, 1))
                    #Adding the loss per batch
                    loss = self.fit_batch(data)
        
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.clip)
                    current_batch += 1
                #TODO: Get the average loss of the whole range and add it to the 
                # loss list
                self.average_batch_loss(current_batch)

            else:
                loss = self.fit_batch(data)
            

            if np.isnan(loss):
                print('Loss is nan!')
                break

            if epoch % self.print_every == 0:
                #We are not printing the average loss across batches, but 
                # the loss on the final batch
                #TODO: print the average loss?
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
        zp = fwd_return['zp']
        kl = 0
        ll = 0

        #For each timestep
        for i in range(len(x)):
            # KL divergence
            kl += self.KL_fn(qzx[i], Normal(0, 1)).sum(1).mean(0)
            ll += pxz[i].log_prob(x[i]).sum(1).mean(0)

        total = kl - ll

        losses = {
            'total': total,
            'kl': kl,
            'll': ll
        }

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

    def average_batch_loss(self, nbatches):
        """
        Remove all teh temporal batches from the loss object
        and add only the average one.
        This way, len(loss) = nepochs.
        (not validated)
        """
        #get avg losses
        avg_losses = {
            'total' : sum(self.loss['total'][-nbatches:]) / nbatches,
            'kl': sum(self.loss['kl'][-nbatches:]) / nbatches,
            'll': sum(self.loss['ll'][-nbatches:]) /  nbatches
        }

        #remove nbatches losses
        for key in self.loss.keys():
            self.loss[key] = self.loss[key][:-nbatches]

        #add avg loss
        for key in self.loss.keys():
            self.loss[key].append(float(avg_losses[key]))


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
