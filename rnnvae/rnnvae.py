"""
File containing the model of the network.

This network is divided in three parts:

Encoder, which from an input x_t and a previous hidden state h_t-1
generates a mu and sigma that generate a latent space z_t.

Decoder, which from a sampled z_t and a hidden state h_t-1 generates
x_hat_t

RNN, that samples from z_t and with x_t and h_t-1 gives a new hidden
state h_t, for the next input.

Those three subnetworks only apply to each of the channels.

The three parts of the network are combined in the class model_MCRNNVAE

For n_channels = 1, results should be the same as a normal network.

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
            #self.phi_layers.append(nn.ReLU())
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

    def __init__(self, input_size, hidden_size, latent_size, n_layers, sigmoid_mean=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.n_layers = n_layers
        self.sigmoid_mean = sigmoid_mean
        #Define module list
        self.init_block()


    def init_block(self):
        self.var_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            # Here we coould add non-linearities if needed
            self.var_layers.append(nn.Linear(self.input_size, self.hidden_size))
            #self.var_layers.append(nn.ReLU())
            self.input_size = self.hidden_size

        #TODO: MU COULD BE USING A SIGMOID FOR THE OUTPUT
        if self.sigmoid_mean:
            self.to_mu = nn.Sequential(
                nn.Linear(self.input_size, self.latent_size),
                nn.Sigmoid()) 
        else:
            self.to_mu = nn.Linear(self.input_size, self.latent_size)
        
        self.to_logvar = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size))   
            # nn.Softplus())
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


class MCRNNVAE(nn.Module):
    """
    This class implements the full model of the network.
    We take as example the models of mcvae, to create it similar to them.
    
    This model can take as input several channels. n_channels=1 has the same funcionality
    sa the original RNN VAE.

    Note: inputs to the network should already be padded, so we should consider
    that all sequences have the same length, be padded accordingly.

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
    :param clip: to clip the gradient
    :param nepochs: number of epochs
    :param batch_size: batch size of the input data, if applicable
    :param n_channels: number of channels of the input data
    :param n_feats: number of features of each channel (tuple)
    :param model_name_dict: name of the different channels
    :param device: device where to run the computations
    :print_every: verbose
    """

    def __init__(self, h_size, phi_x_hidden, phi_x_n, 
                 phi_z_hidden, phi_z_n, enc_hidden, enc_n, latent, dec_hidden, 
                 dec_n, clip, nepochs, batch_size, n_channels, n_feats, model_name_dict, device,
                 print_every=100):

        super().__init__()

        self.dtype = torch.FloatTensor
        self.device = device
        self.n_channels = n_channels
        self.n_feats = n_feats
        self.model_name_dict = model_name_dict

        #Parameters
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
        self.batch_size = batch_size # we only use  this for loss visualization
        self.epochs = nepochs

        #Init KL and loss
        self.optimizer = None
        self.init_KL()
        self.init_loss()

        # init the names (copied from Luigis' code)
        self.init_names()

        # Building blocks
        # Build them for every channel!
        self.ch_priors = nn.ModuleList() # LIST OF PRIORS
        self.ch_phi_x = nn.ModuleList() #LIST OF TRANSFORMATION TO INPUT
        self.ch_enc = nn.ModuleList() #LIST OF ENCODERS
        self.ch_phi_z = nn.ModuleList() #LIST OF TRANSFORMATION TO LATENT
        self.ch_dec = nn.ModuleList() #LIST OF DECODERS
        self.ch_RNN = nn.ModuleList() #LIST OF RNN BLOCK

        for ch in range(self.n_channels): 
            ## PRIOR 
            self.ch_priors.append(VariationalBlock(self.h_size, self.enc_hidden, self.latent, self.enc_n))

            ### ENCODER
            self.ch_phi_x.append(PhiBlock(self.n_feats[ch], self.phi_x_hidden, self.phi_x_n))
            self.ch_enc.append(VariationalBlock(self.phi_x_hidden + self.h_size, self.enc_hidden, self.latent, self.enc_n))

            ### DECODER
            self.ch_phi_z.append(PhiBlock(self.latent, self.phi_z_hidden, self.phi_z_n))
            self.ch_dec.append(VariationalBlock(self.phi_z_hidden + self.h_size, self.dec_hidden, self.n_feats[ch], self.dec_n, sigmoid_mean=False))

            ### RNN 
            self.ch_RNN.append(nn.GRU(self.phi_x_hidden + self.phi_z_hidden, self.h_size, 1))



    def init_names(self):
        """
        Put the names to each channel and feature.
        """
        self.model_name = self._get_name()
        if not self.model_name_dict == None:
            for key in sorted(self.model_name_dict):
                val = self.model_name_dict[key]
                if type(val) == list or type(val) == tuple:
                    val = str(np.sum(val))
                self.model_name += '__' + key + '_' + str(val)

        self.ch_name = ['Ch.' + str(i) for i in range(self.n_channels)]

        self.varname = []
        for ch in range(self.n_channels):
            self.varname.append(['feat.' + str(j) for j in range(self.n_feats[ch])])



	def sample_from(self, qzx):

		'''
		sampling by leveraging on the reparametrization trick
		'''

		zx = []

		for ch in range(self.n_channels):
			if self.training:
				zx.append(
					qzx[ch].rsample()
				)
			else:
				zx.append(
					qzx[ch].loc
				)
		return zx

    def step_sample(self, ht_list):
        """
        Does a single recurrent step, 
        but using sampling from the prior.
        
        ht_list is a list of hidden states, as long as n_channels

        We do not have any x here.
        """
        #We store the results here, for each channel
        x_t_list = []
        ht_list = []
        z_prior_list = []
        z_t_list = []
        dec_t_list = []

        for ch in range(self.n_channels):
            # Sampling from the prior
            ht = ht_list[ch]
            z_prior = self.ch_priors[ch](ht[-1])

            # test this
            z_t = self.sample_from(z_prior)
            # z_t = z_prior.rsample() #Hard sampling because we want variation

            # Apply phi_z
            phi_z_t = self.ch_phi_z[ch](z_t)

            # decoder
            #THIS NEEDS TO BE CHANGED IN SOME WAY
            x = torch.cat([phi_z_t, ht[-1]], 1)
            dec_t_list.append([])
            x_t_list.append([])
            for j in range(self.channels):
                ### DECODER
                # Decode from each different channel
                pxz_t = self.ch_dec[j](x)
                xnext = self.sample_from(pxz_t)
                dec_t_list[ch].append(pxz_t)
                x_t_list[ch].append(xnext)

            # recurrence
            phi_x_t = self.ch_phi_x[ch](x_t)
            _, ht = self.ch_RNN[ch](torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), ht)
            
            #Save the results
            ht_list.append(ht)
            z_prior_list.append(z_prior)
            z_t_list.append(z_t)
        
        return x_t_list, ht_list, z_prior_list, z_t_list, dec_t_list


    def step(self, xt_list, ht_list):
        """
        Function that implements the forward pass 
        of a single recurrent step
        """
        hnext_list = []
        z_prior_list = []
        zx_t_list = []
        qzx_t_list = []
        pxz_t_list = []
        xnext_list = []

        for ch in range(self.n_channels):
            xt = xt_list[ch]
            ht = ht_list[ch]
            ###PRIOR
            z_prior = self.ch_priors[ch](ht[-1])

            ##### ENCODER
            x_phi = self.ch_phi_x[ch](xt)
            # Append both inputs
            x = torch.cat([x_phi, ht[-1]], 1)

            # Run through the encoder
            qzx_t = self.ch_enc[ch](x)
            
            #Sample from
            zx_t = self.sample_from(qzx_t)

            #Apply phi_z
            phi_zx_t = self.ch_phi_z[0](zx_t)

            #Decoder, for each separate channel
            x = torch.cat([phi_zx_t, ht[-1]], 1)
            pxz_t_list.append([])
            xnext_list.append([])
            for j in range(self.channels):
                ### DECODER
                # Decode from each different channel
                pxz_t = self.ch_dec[j](x)
                xnext = pxz_t.loc
                pxz_t_list[ch].append(pxz_t)
                xnext_list[ch].append(xnext)

            ### RNN
            x = torch.cat([phi_zx_t, x_phi],1).unsqueeze(0)
            _, hnext = self.ch_RNN[ch](x, ht)

            hnext_list.append(hnext)
            z_prior_list.append(z_prior)
            zx_t_list.append(zx_t)
            qzx_t_list.append(qzx_t)
        
        return xnext_list, hnext_list, z_prior_list, zx_t_list, qzx_t_list, pxz_t_list

    def sample_latent(self, nsamples, nt):
        """
        Sample a number of samples from a trained latent space
        nsamples: number of generated samples
        nt: number of timepoints
        
        returns a list of nsamples of size (nch x nt x nfeat)
        """
        self.eval()
        x_pred = []
        z = []
        pxz = []
        zp = []

        if self.is_fitted:
            with torch.no_grad():
                #dim0 is the nrecurrentlayers, dim1 is the size of x (we are generating one by one) 
                ht = [Variable(torch.zeros(1, nsamples, self.h_size, device=self.device)) for i in range(self.n_channels)]
                for t in range(nt):
                    x_t, ht, z_prior, z_t, dec_t = self.step_sample(ht)
                    # Get mean across channels
                    x_pred.append([[x2.cpu().detach().numpy() for x2 in x] for x in x_t])
                    z.append([z.cpu().detach().numpy() for z in z_t])
                    pxz.append(dec_t)
                    zp.append(z_prior)
            
            return {
            'xnext' : x_pred,
            'z': z,
            'pxz': pxz,
            'zp': zp
            }

        raise RuntimeError('Model needs to be fit')


    def forward(self, x):
        """
        Forward propagation of the network,
        passing over the full network for every 
        step of the sequence

        So, data shape should be 
        (nt, nch, nbatch, feat)
        """
        # Initial h0
        ht = [Variable(torch.zeros(1, x.size(1), self.h_size, device=self.device)) for i in range(self.n_channels)]
        x_pred = []
        qzx = []
        zx = []
        pxz = []
        zp = []
        # ITERATE OVER THE SEQUENCE
        # and save intermediate results
        for x_t in x:
            xnext, hnext, zp_t, zx_t, qzx_t, pxz_t = self.step(x_t, ht)
            #xnext or x_t?
            x_pred.append(xnext)
            qzx.append(qzx_t)
            zx.append(zx_t)
            pxz.append(pxz_t)
            zp.append(zp_t)
            ht = hnext

        #Return similar to the mcvae implementation
        return {
            'x' : x,
            'xnext': x_pred,
            'qzx': qzx,
            'z': zx,
            'pxz': pxz,
            'zp': zp
        }

    def sequence_predict(self, x, nt):
        """
        Function that predicts, for a given incomplete sequence x,
        return future values for that specific value, over nt timepoints

        We assume that all channels have the same amount of time points and
        they correspond to each other

        x has shape (nt, nch, nbatch, nfeat)
        """
        self.eval()
        if self.is_fitted:
            with torch.no_grad():
                #initalize ht
                ht = [Variable(torch.zeros(1, x.size(1), self.h_size, device=self.device)) for i in range(self.n_channels)]
                #initialize returns
                x_pred = []
                z = []
                pxz = []
                zp = []

                #For each timepoint
                for t in range(nt):
                    # If we have x information, we do a normal forward pass    
                    if t < len(x):
                        x_t = x[t]
                        xnext, hnext, zp_t, z_t, _, pxz_t = self.step(x_t, ht)
                        x_pred.append([[x2.cpu().detach().numpy() for x2 in x] for x in x_t])
                    # if not, we do a sampling from the corresponding prior
                    else:
                        xnext, hnext, zp_t, z_t, pxz_t = self.step_sample(ht)
                        x_pred.append([[x2.cpu().detach().numpy() for x2 in x] for x in xnext])
                    # set the next x (if we have it or not)
                    z.append(z_t)
                    pxz.append(pxz_t)
                    zp.append(zp_t)
                    ht = hnext
                
                seq_pred = {
                    'xnext': x_pred,
                    'z': z,
                    'pxz': pxz,
                    'zp': zp
                }

                # Convert to numpy the xnext and the zx values
                # it is a double loop, for each time point and each channel (not sure is really efficient)
                seq_pred['z'] = np.array([[x2.cpu().detach().numpy() for x2 in x] for x in seq_pred["z"]])
                return seq_pred

        raise RuntimeError('Model needs to be fit')


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


    def predict(self, data):
        """
        Predict the reconstruction of some input.
        We are predicting the exact reconstuction, and its latent space
        """
        self.eval()
        
        if self.is_fitted:
            with torch.no_grad():
                pred = self.forward(data)

                # Convert to numpy the xnext and the zx values
                pred['xnext'] = np.array([[x2.cpu().detach().numpy() for x2 in x] for x in pred["xnext"]])
                pred['z'] = np.array([[x2.cpu().detach().numpy() for x2 in x] for x in pred["z"]])
                return pred
        
        raise RuntimeError('Model needs to be fit')

    def reconstruct(self, x, nt, reconstruct_from=None):
        """
        Reconstruct the other channels from available channels.

        We use the channels indicated in reconstruct_from to use the data from x to reconstruct
        for nt times.

        Return a list of nt elements, each element is a list of ch elements, etc etc
        """
		available_channels = range(self.n_channels) if reconstruct_from is None else reconstruct_from
		fwd_return = self.forward(x)
		pxz = fwd_return['pxz']

		Xhat = []
        for t in range(nt):
            Xhat.append([])
            for c in range(self.n_channels):
                # mean along the stacking direction
                xhat = torch.stack([pxz[e][c].loc.cpu().detach() for e in available_channels]).mean(0)
                Xhat[t].append(xhat.numpy())
                del xhat

		return Xhat


    def fit(self, data_train, data_val):
        """
        Optimize full training.
        
        Each epoch, we optimize over the data_train. After optimizing, we evaluate
        over unseen data.

        Data is of the form
        (nt, nch, nbatch, feat)
        """

        for epoch in range(self.epochs):

            ## TRAINING
            self.train()  # Inherited method which sets self.training = True
            if type(data_train) is torch.utils.data.DataLoader:
                #If we have a data loader
                ##WE WILL HAVE A DATA LOADER,
                # MAKE SURE THAT THE FORMAT IS CORRECT
                current_batch = 0
                for local_batch in data_train:
                    #happens if the dataset has labels, discard them
                    if type(local_batch) is list:
                        local_batch = local_batch[0]
                    # Study well why this transpose and everything
                    #Will probably be removed when using real data
                    #Adding the loss per batch      
                    loss = self.fit_batch(data_train)
        
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.clip)
                    current_batch += 1

                self.loss = self.average_batch_loss(current_batch, self.loss)

            else:
                loss = self.fit_batch(data_train)
            
            # Check loss nan
            if np.isnan(loss):
                print('Loss is nan!')
                break

            ##VALIDATION
            self.eval()  # Inherited method which sets self.training = False
            with torch.no_grad():
                if type(data_val) is torch.utils.data.DataLoader:
                    #If we have a data loader
                    ##WE WILL HAVE A DATA LOADER,
                    # MAKE SURE THAT THE FORMAT IS CORRECT
                    current_batch = 0
                    for local_batch in data_val:
                        #happens if the dataset has labels, discard them
                        if type(local_batch) is list:
                            local_batch = local_batch[0]
                        # Study well why this transpose and everything
                        #Will probably be removed when using real data
                        #Adding the loss per batch
                        pred = self.forward(data_val)
                        loss = self.loss_function(pred)

                        current_batch += 1

                    self.val_loss = self.average_batch_loss(current_batch, self.val_loss)

                else:
                    pred = self.forward(data_val)
                    loss = self.loss_function(pred)
                    self.val_loss = self.save_loss(loss, self.val_loss)

            if epoch % self.print_every == 0:
                #We are not printing the average loss across batches, but 
                # the loss on the final batch
                #TODO: print the average loss?
                print('Train loss', end = ' ')
                self.print_loss(epoch, self.loss)
                print('Validation loss', end = ' ')
                self.print_loss(epoch, self.val_loss)
                if loss_has_diverged(self.loss['total']):
                    print('Loss diverged!')
                    break

        self.eval()  # Inherited method which sets self.training = False
        self.is_fitted = True

    def init_KL(self):
        # KL divergence from Pytorch
        self.KL_fn = kl_divergence


    def loss_function(self, fwd_return):
        """
        Full loss function, as described in the paper.

        This is a multi-channel loss: we compute all the losses
        across the different channels.
        """
        x = fwd_return['x']
        qzx = fwd_return['qzx']
        pxz = fwd_return['pxz']
        zp = fwd_return['zp']
        kl = 0
        ll = 0


        # For each time point,
        for t in range(len(x)):

            for i in range(self.n_channels):
                # KL divergence
                #the second distribution is not the normal, is the prior!!
                kl += self.KL_fn(qzx[t][i], zp[t][i]).sum(1).mean(0)
                for j in range(self.n_channels):
                    # i = latent comp; j = decoder
                    # Direct (i=j) and Crossed (i!=j) Log-Likelihood
                    ll += pxz[t][i][j].log_prob(x[t][j]).sum(1).mean(0)

        total = kl - ll

        losses = {
            'total': total,
            'kl': kl,
            'll': ll
        }

        if self.training:
            self.loss = self.save_loss(losses, self.loss)
            return total
        else:
            return losses

    def init_loss(self):
        self.loss = {
            'total': [],
            'kl': [],
            'll': []
        }
        self.val_loss = {
            'total': [],
            'kl': [],
            'll': []
        }

    def average_batch_loss(self, nbatches, loss):
        """
        Remove all teh temporal batches from the loss object
        and add only the average one.
        This way, len(loss) = nepochs.
        (not validated)
        """
        #get avg losses
        avg_losses = {
            'total' : sum(loss['total'][-nbatches:]) / nbatches,
            'kl': sum(loss['kl'][-nbatches:]) / nbatches,
            'll': sum(loss['ll'][-nbatches:]) /  nbatches
        }

        #remove nbatches losses
        for key in loss.keys():
            loss[key] = loss[key][:-nbatches]

        #add avg loss
        for key in loss.keys():
            loss[key].append(float(avg_losses[key]))
        return loss

    def print_loss(self, epoch, loss):
        print('Epoch: {:4d}/{} ({:.0f}%)\tLoss: {:.4f}\tLL: {:.4f}\tKL: {:.4f}\tLL/KL: {:.4f}'.format(
            epoch,
            self.epochs,
            100. * (epoch) / self.epochs,
            loss['total'][-1],
            loss['ll'][-1],
            loss['kl'][-1],
            loss['ll'][-1] / (1e-8 + loss['kl'][-1])
        ), end='\n')

    def save_loss(self, new_losses, loss):
        for key in loss.keys():
            loss[key].append(float(new_losses[key].detach().item()))
        return loss

    def save(self, out_dir, file_name):
        """
        Pickles the model parameters to be retrieved later

        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        path = out_dir + '/' + file_name
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        torch.save(self.state_dict(), path)

    def load(self, model_path):
        """
        Loads the model's parameters from the path mentioned

        :param model_path: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(model_path))