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

SPECIFICATIONS OF THIS VERSION:
- NO PHI_Z OR PHI_X
EVOLVED FROM RNNVAE_H (SO SHARED RNN, SHARED Z)
"""

import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence, Categorical, MultivariateNormal, Bernoulli
import os

# Functions
def reconstruction_error(predicted, target, mask):
    #Full reconstruction error.
    # sum over data dimensions (n_feats); average over observations (N_obs)
    #Mask contains the subjects upon which we do not need to sum over
    rec_aux = ((target - predicted) ** 2).sum(1)
    rec_masked = torch.masked_select(rec_aux, mask)
    return rec_masked.mean(0)

def compute_ll(dist, x):
    """
    Function to compute the log likelihood 
    depending on the output distribution.
    """
    if isinstance(dist, Normal):
        ll = dist.log_prob(x).mean(1, keepdims=True)
    elif isinstance(dist, Categorical):
        ll = dist.log_prob(x.view(-1)).unsqueeze(1)
    elif isinstance(dist, MultivariateNormal):
        ll = dist.log_prob(x).unsqueeze(1)  # MultiVariate already sums over dimensions
    else:
        raise NotImplementedError

    return ll

def KL_log_uniform(p, *args, **kwargs):
    """
    Arguments other than 'p' are ignored.

    Formula from Paragraph 4.2 in:
    Variational Dropout Sparsifies Deep Neural Networks
    Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
    https://arxiv.org/abs/1701.05369
    """
    log_alpha = compute_log_alpha(p.loc, p.scale.pow(2).log())
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    neg_KL = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) - k1
    return -neg_KL

def mae(predicted, target, mask):
    """
    Mean Absolute Error
    """
    # sum over data dimensions (n_feats); average over observations (N_obs)
    mae_aux = torch.abs(target - predicted).sum(1)
    mae_masked = torch.masked_select(mae_aux, mask)
    return mae_masked.mean(0)  # torch.Size([1])

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

    def __init__(self, input_size, hidden_size, latent_size, n_layers, sigmoid_mean=False, log_alpha=None, c_z=None, decode=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.n_layers = n_layers
        self.sigmoid_mean = sigmoid_mean
        self.log_alpha = log_alpha
        self.c_z = c_z
        if self.c_z is None:
            self.c_z = input_size 
        #Define module list

        self.decode=decode # Decoder flag, if True, variance param shared

        self.init_block()


    def init_block(self):
        self.var_layers = nn.ModuleList()
        if self.c_z < self.input_size:
            self.input_size = self.c_z
        for _ in range(self.n_layers):
            # Here we coould add non-linearities if needed
            self.var_layers.append(nn.Linear(self.input_size, self.hidden_size))
            self.var_layers.append(nn.ReLU())
            self.input_size = self.hidden_size

        if self.sigmoid_mean:
            self.to_mu = nn.Sequential(
                nn.Linear(self.input_size, self.latent_size),
                nn.Sigmoid()) 
        else:
            self.to_mu = nn.Linear(self.input_size, self.latent_size)
        
        if self.log_alpha is None:
            if self.decode:
                noise_init_logvar=-3 ## HARDCODED
                tmp_noise_par = torch.FloatTensor(1, self.latent_size).fill_(noise_init_logvar)
                self.to_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=True)
            else:
                self.to_logvar = nn.Sequential(
                    nn.Linear(self.input_size, self.latent_size))   
            #nn.Softplus())
        #Last layers, to obtain from the same tal both outputs  


    def forward(self, x):
        x = x[:,:self.c_z] #Select only the first x dimensions

        for i in range(len(self.var_layers)):
            x = self.var_layers[i](x)
        mu = self.to_mu(x)
        if self.log_alpha is None:
            if self.decode:
                logvar = self.to_logvar.exp().pow(0.5)# + 1e-3#
            else:
                logvar = self.to_logvar(x).exp().pow(0.5)# + 1e-3#
        else:
            logvar = compute_logvar(mu, self.log_alpha).exp().pow(0.5)

        x = Normal(
            loc=mu,
            scale=logvar
        )
        return x

class DecoderBernoulli(nn.Module):
    """
    Class implementing a decoder with a Bernoulli distribution, for the
    0/1 class (sex, apoe)
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers, sigmoid_mean=False, log_alpha=None, c_z=None, decode=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.sigmoid_mean = sigmoid_mean
        self.log_alpha = log_alpha
        self.c_z = c_z
        if self.c_z is None:
            self.c_z = input_size 
        #Define module list

        self.decode=decode # Decoder flag, if True, variance param shared
        self.init_block()

    def init_block(self):
        print('nyi')

    def forward(self):
        print('nyi')

class DecoderCat(nn.Module):
    """
    Class implementing a decoder with a Categorical distribution for, categorical
    biomarkers (APOE)
    """
    def __init__(self, input_size, hidden_size, n_cat, n_layers, c_z=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_cat = n_cat
        self.n_layers = n_layers
        self.c_z = c_z
        if self.c_z is None:
            self.c_z = input_size 

        self.init_block()

    def init_block(self):
        self.var_layers = nn.ModuleList()
        if self.c_z < self.input_size:
            self.input_size = self.c_z

        for _ in range(self.n_layers):
            # Here we coould add non-linearities if needed
            self.var_layers.append(nn.Linear(self.input_size, self.hidden_size))
            self.var_layers.append(nn.LeakyReLU())
            self.input_size = self.hidden_size

        self.output = nn.Linear(self.input_size, self.n_cat)
    
    def forward(self, x):
        x = x[:,:self.c_z] #Select only the first x dimensions

        for i in range(len(self.var_layers)):
            x = self.var_layers[i](x)
        out = self.output(x)
        x = Categorical(logits=out)
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
    :param ch_type: list with either "long" or "bL", depending on data being
    : cross sectional or not
    :param n_feats: number of features of each channel (tuple)
    :param c_z: list with ints, for each channel, indicating the number of dimensions in z to use (must be less than latent) 
    :param device: device where to run the computations
    :param ch_dist: None, if all ch have normal dist, or list of types of dist. Can be "Normal", "Categorical" or ""
    :param model_name_dict: name of the different channels
    :print_every: verbose
    :phi_layers: whether to use phi layers or not
    :sigmoid_mean: apply sigmoid to the mean of the decoder/encoder.
    :param dropout: parameter that says if there is a dropout or not
    :param dropout_threshold: threshold onto apply dropout onto the latent spaces 
    """

    def __init__(self, h_size, enc_hidden, enc_n, latent, dec_hidden, 
                 dec_n, clip, nepochs, batch_size, n_channels, ch_type,
                 n_feats, c_z, device, ch_dist=None,
                 model_name_dict=None, print_every=100, phi_layers=True, 
                 sigmoid_mean=False, dropout=False, dropout_threshold=0.2):

        super().__init__()

        self.dtype = torch.FloatTensor
        self.device = device
        self.n_channels = n_channels
        self.n_feats = n_feats
        self.model_name_dict = model_name_dict
        self.ch_type = ch_type

        #Parameters
        self.h_size = h_size
        self.enc_hidden = enc_hidden
        self.enc_n = enc_n #n layers
        self.latent = latent
        self.dec_hidden = enc_hidden
        self.dec_n = enc_n
        self.dropout = dropout
        self.dropout_threshold = dropout_threshold #dropout threshold
        self.c_z = c_z
        
        # other parameters
        self.clip = clip
        self.print_every = print_every
        self.batch_size = batch_size # we only use  this for loss visualization
        self.epochs = nepochs
        self.curr_epoch = 0
        self.reduced_training = 0 ##NUMBER OF EPOCHS UNTIL WE WILL MULTIPLY THE STD BY 0.1 
        self.phi_layers = phi_layers #if we use the phi functions or not 
        self.sigmoid_mean = sigmoid_mean # if we use sigmoid for the mean in the decoder and encoder
        self.sampling=False

        # init the names (copied from Luigis' code)
        self.init_names()

        # Building blocks
        # Build them for every channel!
        self.ch_priors = VariationalBlock(self.h_size, self.enc_hidden, self.latent, self.enc_n)#nn.ModuleList() # LIST OF PRIORS
        self.ch_enc = nn.ModuleList() #LIST OF ENCODERS

        self.ch_dec = nn.ModuleList() #LIST OF DECODERS
        self.ch_RNN = nn.RNN(self.latent, self.h_size, 1)

        # if dropout, intiate the log alpha
        self.log_alpha = None
        if self.dropout:
            self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.latent).normal_(0, 0.01))

        for ch in range(self.n_channels): 
                
            #########################################
            # Define the layers for each channel. For channels that are baseline, the channels involving H should change the inpu size and 
            if self.ch_type[ch] == 'bl':
                self.dec_input = self.latent
                self.enc_input = self.n_feats[ch]
            else:
                self.dec_input = self.latent + self.h_size
                self.enc_input = self.n_feats[ch] + self.h_size

            ### ENCODER
            self.ch_enc.append(VariationalBlock(self.enc_input, self.enc_hidden, self.latent, self.enc_n, sigmoid_mean=self.sigmoid_mean, log_alpha=self.log_alpha))

            ### DECODER
            #if ch == 3:
            #    self.ch_dec.append(DecoderCat(self.dec_input, self.dec_hidden, 3, self.dec_n, c_z=self.c_z[ch]))
            #else:
            self.ch_dec.append(VariationalBlock(self.dec_input, self.dec_hidden, self.n_feats[ch], self.dec_n, sigmoid_mean=self.sigmoid_mean, c_z=self.c_z[ch], decode=True))

        #Init KL and loss
        self.optimizer = None
        self.init_KL()
        self.init_loss()


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

    @staticmethod
    def p_to_prediction(p):

        if isinstance(p, list):
            return [MCRNNVAE.p_to_prediction(_) for _ in p]

        if isinstance(p, Normal):
            pred = p.loc
        elif isinstance(p, Categorical):
            pred = p.logits.argmax(dim=1).unsqueeze(1).float()
        elif isinstance(p, Bernoulli):
            pred = p.probs
        else:
            raise NotImplementedError

        return pred


    def sample_from(self, qzx):

        '''
        sampling by leveraging on the reparametrization trick
        '''
        if self.training or self.sampling:
            zx = qzx.rsample()
        else:
            zx = MCRNNVAE.p_to_prediction(qzx)
        return zx

    def dropout_fn(self, lv_ch):
        # Dropout function.
        #Apply a variational dropout to each channel.
        #a single channel
        alpha = torch.exp(self.log_alpha.detach())
        do = alpha / (alpha + 1)

        lv_out = lv_ch * (do < self.dropout_threshold).float()
        return lv_out

    def step_predict(self, xt_list, ht_list, av_ch=None):
        """
        Step for the function predict, that takes into account
        the possible available or not channel at a single time step.

        Depending on the av_ch, the channels that are indicated there
        are predicted as usual, and the others are inferred from prior 
        and other channels.

        av_ch should be the same size as xt_list and correspond to 
        the channels included in xt_list. Ht list should be the full
        list
        """
        av_ch = self.n_channels if av_ch is None else av_ch
        hnext_list = []
        z_prior_list = []
        zx_t_list = []
        qzx_t_list = []
        pxz_t_list = []
        zx_list = []
        x_list = []

        #ht_prior = torch.stack(ht_list).mean(0)
        #z_prior = self.ch_priors(ht_prior[-1]) # get the prior from the hidden state

        #For each single channel
        for i in range(self.n_channels):
            ht = ht_list[i]
            z_prior = self.ch_priors(ht[-1]) # get the prior from the hidden state
            if i in av_ch:
                #If we have available channel, do normal pipeline 
                xt = xt_list[av_ch.index(i)]
                x_list.append(xt)

                if self.ch_type[i] == "long": #if its not longitudinal, do not use ht
                    x = torch.cat([xt, ht[-1]], 1) # append input with hidden
                else:
                    x = xt

                qzx_t = self.ch_enc[i](x) # Run through the encoder
                z_t = self.sample_from(qzx_t) #Sample from q(z|x)
                qzx_t_list.append(qzx_t)
            else:
                z_t = self.sample_from(z_prior) #sample from the prior
                qzx_t_list.append(None) #To solve mistakes, but not used in sampling (no encoder!)
                x_list.append(None) #Not used in samples

            ###DROPOUT
            if not self.training and self.dropout:
                z_t = self.dropout_fn(z_t)  

            zx_list.append(z_t)  # In the sampling moment, it is necessary

            #Decoder, for each separate channel
            pxz_t_list.append([])
    
            for j in range(self.n_channels):
                # if its not longitudinal, do not use ht
                if self.ch_type[j] == "long":
                    z = torch.cat([z_t, ht[-1]], 1)
                else:
                    z = z_t
                pxz_t = self.ch_dec[j](z)  # Decode from each different channel
                pxz_t_list[i].append(pxz_t)

            z_prior_list.append(z_prior)
            zx_t_list.append(z_t)

        # Second part: generate the samples if needed, we needed to generate
        # all the p_xz 
        xhat_list = []
        for i in range(self.n_channels):
            # THIS HAS BEEN HARD COMMENTED, PROBABLY NEEDS TO BE THE COMBINATION
            # TRAIN WITH THE OTHER VERSION?
            x_hat = torch.stack([MCRNNVAE.p_to_prediction(pxz_t_list[e][i]) for e in range(self.n_channels)]).mean(0)
            #x_hat = pxz_t_list[i][i].loc
            xhat_list.append(x_hat)

            if self.ch_type[i] == 'long':
                _, hnext = self.ch_RNN(zx_list[i].unsqueeze(0), ht) # Recurrence step
            else:
                # Just put a dummy information, will not be used
                hnext = ht

            #Save results
            hnext_list.append(hnext)

        ###DO CATEGORICAL APPROXIMATION HERE


        # Return xt_list, in the not sampling case, this does nothing, in the other case, returns the reconstructed x
        return xhat_list, hnext_list, z_prior_list, zx_t_list, qzx_t_list, pxz_t_list


    def predict(self, data, mask, nt, av_ch=None, task='prediction'):
        """
        Predict the reconstruction of some input.
        av_ch is the channels that we will be using for reconstruction. If its None, we just use all of them
        We are predicting the exact reconstuction, and its latent space

        The variable task can be either "prediction" or "reconstruction".
        If "prediction", that means that we are predicting future timepoints
        for available data: this means that each extra timepoint we need to predict will come
        only from the previous pxz of the same channel.
        If it is reconstruction, then we will reconstruct from the channels indicated by
        av_ch.
        """
        self.eval()
        av_ch = range(self.n_channels) if av_ch is None else av_ch

        #Again, això està malament. channels poden tenir differnet num de tps!
        # data_ntp = data[0].size(0)
        if self.is_fitted:
            with torch.no_grad():
                # initalize ht
                # all channels have same number of samples, so x[0].size(1) can be hardcoded
                ht = [Variable(torch.zeros(1, data[0].size(1), self.h_size, device=self.device)) for _ in range(self.n_channels)]
                # initialize returns
                z = [[] for _ in range(self.n_channels)]
                pxz = [[] for _ in range(self.n_channels)]
                zp = [[] for _ in range(self.n_channels)]
                qzx = [[] for _ in range(self.n_channels)]

                #For each timepoint
                for tp in range(nt):
                    #Need to repopulate the list so that it contains a single time point, still being a list
                    #If we sample, we just use the
                    curr_channels = [i for i, x_ch in enumerate(data) if tp < x_ch.shape[0]]
                    x_t = [x_ch[tp, :, :] for i, x_ch in enumerate(data) if i in curr_channels]
                    mask_i = [mask_ch[tp, :, :] for i, mask_ch in enumerate(mask) if i in curr_channels]
                    #if we have already done baseline:
                    if tp > 0:
                        ##TODO POTENTIAL OF PROBLEMS HERE, MAKE SURE THAT IT WORKS WELL
                        # fill missing values in input data with the predicted xhat
                        # mask_i and x_t are the same length as curr_channels, are correspond to each other
                        # xhat no, has all the channels
                        x_t = [torch.where(mask_i[i], x_t[i], xhat[ch]) for i, ch in enumerate(curr_channels)]

                    # If we have x information, we do a normal step
                    xhat, hnext, zp_t, z_t, qzx_t, pxz_t = self.step_predict(x_t, ht, curr_channels)
                    # zp.append(zp_t)
                    for i in range(self.n_channels):
                        #this xnext needs to be averaged across all the values, as it is reocnstructed from all channels
                        z[i].append(z_t[i])
                        pxz[i].append(pxz_t[i])
                        zp[i].append(zp_t[i])
                        qzx[i].append(qzx_t[i])
                    ht = hnext

                # For xnext, we need to obtain the average of the reconstruction across channels and timepoints
                
                X_hat = [[] for _ in range(self.n_channels)]
                for i in range(self.n_channels):
                    for tp in range(nt):
                        # this is not adequate. We should always use the various channels to predict a single time point.
                        if task=='prediction':
                            xhat = MCRNNVAE.p_to_prediction(pxz[i][tp][i]).cpu().detach()
                            # xhat = torch.stack([pxz[ch][tp][i].loc.cpu().detach() for ch in av_ch]).mean(0)

                        elif task=='recon':
                            #select only longitudinal for tp > 0?
                            #if tp > 0:
                            #    xhat = torch.stack([MCRNNVAE.p_to_prediction(pxz[ch][tp][i]).cpu().detach() for ch in av_ch if self.ch_type[ch] == 'long']).mean(0)
                            #else:
                            xhat = torch.stack([MCRNNVAE.p_to_prediction(pxz[ch][tp][i]).cpu().detach() for ch in av_ch]).mean(0)
                        #else:
                        #    print('wrong!')
                        #    xhat = [None]
                        # curr_channels = [i for i, x_ch in enumerate(data) if tp < x_ch.shape[0]]
                        # 

                        ### DO CATEGORICAL APPROXIMATION HERE

                        X_hat[i].append(xhat.numpy())
                    X_hat[i] = np.asarray(X_hat[i]) #Convert to np array

                z = np.array([[x2.cpu().detach().numpy() for x2 in x] for x in z])

                pred = {
                    'xnext': X_hat,
                    'z': z,
                    'qzx': qzx,
                    'pxz': pxz,
                    'zp': zp
                }
                return pred
        
        raise RuntimeError('Model needs to be fit')



        # The channels that are not available
    def step(self, xt_list, ht_list, curr_channels, av_ch=None):
        """
        Function that implements the forward pass 
        of a single recurrent step for all the channels.
        curr_channels contains the number of the channels contained in xt_list

        ht_list is a list for all the available channels, of the hidden states
        xt_lsit is the list of data, one per time channel
        curr_channels, ht_list and xt_list should have the same length
        """
        av_ch = curr_channels if av_ch is None else av_ch
        hnext_list = []
        z_prior_list = []
        zx_t_list = []
        qzx_t_list = []
        pxz_t_list = []
        zx_list = []

        # First part: get all the decoders
        #Assume that the n_channels in the input can be variable
        for i in range(len(curr_channels)):
            ch = curr_channels[i]
            ht = ht_list[i]
            z_prior = self.ch_priors(ht[-1]) # get the prior from the hidden state
            if self.curr_epoch < self.reduced_training:
                z_prior.scale *= 0.1

            #If we are not sampling from the prior, we have an input value
            xt = xt_list[i]

            if self.ch_type[ch] == "long": #if its not longitudinal, do not use ht
                x = torch.cat([xt, ht[-1]], 1) # append input with hidden
            else:
                x = xt
            qzx_t = self.ch_enc[ch](x)

            if self.curr_epoch < self.reduced_training:
                qzx_t.scale *= 0.1

            z_t = self.sample_from(qzx_t) #Sample from q(z|x)
            qzx_t_list.append(qzx_t)

            ###DROPOUT
            if not self.training and self.dropout:
                z_t = self.dropout_fn(z_t)

            #Apply phi_z
            zx_list.append(z_t)  # In the sampling moment, it is necessary

            #Decoder, for each separate channel 
            pxz_t_list.append([])
            #Decode from each other channel
            for j in range(self.n_channels):
                #If the channel decoder is from baseline, we use only phi_zx_t, if not, add ht
                # if its not longitudinal, do not use ht
                if self.ch_type[j] == "long":
                    z = torch.cat([z_t, ht[-1]], 1)
                else:
                    z = z_t
                pxz_t = self.ch_dec[j](z)  # Decode from each different channel
                if self.curr_epoch < self.reduced_training:
                    pxz_t.scale *= 0.1

                pxz_t_list[i].append(pxz_t)

            #append prior and sample
            z_prior_list.append(z_prior)
            zx_t_list.append(z_t)

            if self.ch_type[ch] == 'long':
                _, hnext = self.ch_RNN(z_t.unsqueeze(0), ht) # Recurrence step
            else:
                # Just put a dummy information, will not be used
                hnext = ht
            #Save results
            hnext_list.append(hnext)

        # Return xt_list, in the not sampling case, this does nothing, in the other case, returns the reconstructed x
        return hnext_list, z_prior_list, zx_t_list, qzx_t_list, pxz_t_list

    def forward(self, x):
        """
        Forward propagation of the network,
        passing over the full network for every 
        step of the sequence

        So, data shape should is a list of the channels, each list a tensor
        input is
        (nch nt nbatch nfeat)
        output should also be that
        (n_ch nt, nbatch, feat)
        """
        # Initial h0
        #all channels have same number of samples, so x[0].size(1) can be hardcoded
        ht = [Variable(torch.zeros(1, x[0].size(1), self.h_size, device=self.device)) for _ in range(self.n_channels)]
        #Initialize output as number of channels
        qzx = [[] for _ in range(self.n_channels)]
        zx = [[] for _ in range(self.n_channels)]
        pxz = [[] for _ in range(self.n_channels)]
        zp = [[] for _ in range(self.n_channels)]

        # ITERATE OVER THE SEQUENCE, at each time point
        #MISTAKE: HERE WE ALWAYS ITERATE, AT MAX, FROM THE NTP FROM CHANNEL 0. SHOULD BE FROM THE MAX
        # OR AT LEAST FORCE ALL TPS TO HAVE THE SAME SIZE
        curr_channels = list(range(self.n_channels))
        tp = 0
        #Mentre hi hagi canals uqe encara tinguin tps
        while len(curr_channels) > 0:
            #Select only channels that have that tp in their data
            curr_channels = [i for i, x_ch in enumerate(x) if tp < x_ch.shape[0]]
            x_t = [x_ch[tp, :, :] for i, x_ch in enumerate(x) if i in curr_channels]
            h_t = [hti for i, hti in enumerate(ht) if i in curr_channels]
            tp += 1
            #Only run those channels
            if len(curr_channels) == 0: break
            hnext, zp_t, zx_t, qzx_t, pxz_t = self.step(x_t, h_t, curr_channels)
            #Recover the initial shape. Append each channel where it corresponds
            # this is for eacah channel
            #but wait, are we appending it where it corresponds??
            # zp.append(zp_t)
            for i in range(len(x_t)):
                ch = curr_channels[i]
                zx[ch].append(zx_t[i])
                pxz[ch].append(pxz_t[i])
                zp[ch].append(zp_t[i])
                qzx[ch].append(qzx_t[i])
                ht[ch] = hnext[i]

        return {
            'x' : x,
            'qzx': qzx,
            'z': zx,
            'pxz': pxz,
            'zp': zp
        }

    def fit_batch(self, x_batch, mask):
        """
        Function to optimize a batch of sequences.
        """
        #with torch.autograd.detect_anomaly():
        pred = self.forward(x_batch)
        loss = self.loss_function(pred, mask)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.clip)
        self.optimizer.step()
        return loss.detach().item()



    def fit(self, data_train, data_val, mask_train=None, mask_val=None):
        """
        Optimize full training.
        
        Each epoch, we optimize over the data_train. After optimizing, we evaluate
        over unseen data.

        Data is of the form
        (nt, nch, nbatch, feat)

        The mask indicates, for each channel, the existence of that sample or not, to take it into
        account when computing the loss.
        """
        if mask_train is None:
            mask_train = [torch.ones(x.shape, dtype=torch.bool).to(self.device) for x in data_train]
        if mask_val is None:
            mask_val = [torch.ones(x.shape, dtype=torch.bool).to(self.device) for x in data_val]

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
                    loss = self.fit_batch(data_train, mask_train)
        
                    current_batch += 1

                self.loss = self.average_batch_loss(current_batch, self.loss)

            else:
                loss = self.fit_batch(data_train, mask_train)
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
                        loss = self.loss_function(pred, mask_val)

                        current_batch += 1

                    self.val_loss = self.average_batch_loss(current_batch, self.val_loss)
                else:
                    pred = self.forward(data_val)
                    loss = self.loss_function(pred, mask_val)
                    self.val_loss = self.save_loss(loss, self.val_loss)

            if epoch % self.print_every == 0:
                #We are not printing the average loss across batches, but 
                # the loss on the final batch
                #TODO: print the average loss?
                print('Train loss', end = ' ')
                self.print_loss(epoch, self.loss)
                print('Validation loss', end = ' ')
                self.print_loss(epoch, self.val_loss)
                if self.dropout:
                    print(self.dropout_comp)
                if loss_has_diverged(self.loss['total']):
                    print('Loss diverged!')
                    break
            self.curr_epoch += 1

        self.eval()  # Inherited method which sets self.training = False
        self.is_fitted = True

    def init_KL(self):
        # KL divergence from Pytorch
        self.KL_fn = kl_divergence


    def loss_function(self, fwd_return, mask=None):
        """
        Full loss function, as described in the paper.

        This is a multi-channel loss: we compute all the losses
        across the different channels.

        The mask should have the same shape as x
        """
        X = fwd_return['x']
        qzx = fwd_return['qzx']
        pxz = fwd_return['pxz']
        zp = fwd_return['zp']
        kl = 0
        ll = 0
        kl_ch = {}
        for ch in range(self.n_channels):
            kl_ch[ch] = 0
            kl_ch[f"{ch}_n"] = 0

        # Need to compute the number of timepoints of each subject at each channel
        # in order to normalize later by number of timepoints
        #this probably would work better in a function to compute it only once
        ntp_subj_list = []
        for ch in range(self.n_channels):
            mask_ch = mask[ch]
            # sum amount of true for every subject
            # and compute the number we will multiply each loss value to normalise
            n_tp_mask = torch.sum(mask_ch[:,:,0], dim=0, dtype=torch.float32).pow_(-1)
            ntp_subj_list.append(n_tp_mask)        

        curr_channels = list(range(self.n_channels))
        t = 0
        #Mentre hi hagi canals uqe encara tinguin tps
        while len(curr_channels) > 0:
            #Select only channels that have that tp in their data
            curr_channels = [i for i, x_ch in enumerate(X) if t < x_ch.shape[0]]
            x = [x_ch[t, :, :] for i, x_ch in enumerate(X) if i in curr_channels]
            mask_i = [mask_ch[t, :, 0] for i, mask_ch in enumerate(mask) if i in curr_channels]
            for i in range(len(curr_channels)):
                ch = curr_channels[i]

                # TO ACCOUNT FOR DIFFERENT NUMBER OF FEATURES
                # INSTEAD OF SUM(1), USE MEAN(1)
                if t == 0 and self.dropout:
                # if its baseline, even if its not t=0, we can approximate it, because no recurrent step
                #if (t == 0  or self.ch_type[ch] == 'bl') and self.dropout:
                    kl_base = KL_log_uniform(qzx[ch][t]).mean(1, keepdims=True)
                #elif t == 0:
                #    kl_base = self.KL_fn(qzx[ch][t], Normal(0, 1)).sum(1)
                else:
                    kl_base = self.KL_fn(qzx[ch][t], zp[ch][t]).mean(1, keepdims=True)
                # if the mask evaluates to zero, it means that, for this time point, this channel doesnt exist. Ignore it.
                if not torch.sum(mask_i[i]) == 0:
                    #kl_base = kl_base * ntp_subj_list[ch] # apply the cross-subject mean
                    kl_masked = torch.masked_select(kl_base, mask_i[i]) # apply the timepoint mask
                    kl += kl_masked.mean(0)
                    kl_ch[ch] += kl_masked.mean(0)
                    kl_ch[f"{ch}_n"] += 1

                for j in range(len(curr_channels)):
                    ch2 = curr_channels[j]
                    # ch = latent comp; ch2 = decoder
                    # Direct (ch=ch2) and Crossed (ch!=ch2) Log-Likelihood
                    #ll_base = -0.5 * ((pxz[ch][t][ch2].loc - x[j]) ** 2 / pxz[ch][t][ch2].scale.exp() + pxz[ch][t][ch2].scale + torch.log(2 * torch.FloatTensor([np.pi]).to(self.device)))
                    #ll_base = ll_base.mean(1)
                    ll_base = compute_ll(pxz[ch][t][ch2], x[j])

                    # ll_base = pxz[ch][t][ch2].log_prob(x[j]).mean(1, keepdims=True)
                    #loss = nn.MSELoss(reduce=False)
                    #ll_base = loss(pxz[ch][t][ch2].loc, x[j]).mean(1)
                    # compute the combined mask for both inputs
                    combined_mask = torch.logical_and(mask_i[i], mask_i[j])
                    #Compute the combined value for the mean between both (its the maximum)
                    #subj_mean = torch.max(ntp_subj_list[ch], ntp_subj_list[ch2])
                    
                    # if the mask evaluates to zero, it means that, for this time point, this channel doesnt exist. Ignore it.
                    if not torch.sum(combined_mask) == 0:
                        #ll_base = ll_base*subj_mean
                        ll_masked = torch.masked_select(ll_base, combined_mask)
                        ll += ll_masked.mean(0)

            t += 1

        #Sum the weighted kl
        #for ch in range(self.n_channels):
        #    kl += kl_ch[ch] / kl_ch[f"{ch}_n"]
        #if self.curr_epoch == 200:
        #    import pdb; pdb.set_trace()
        total = kl - ll

        if np.isnan(total.detach().item()):
            print("loss is nan!")
            import pdb; pdb.set_trace()

        losses = {
            'total': total,
            'kl': kl,
            'll': ll
        }

        if self.training:
            self.loss = self.save_loss(losses, self.loss)
            for i in range(self.n_channels):
                self.kl_loss[i].append(kl_ch[i])

            return total
        else:
            return losses

    ##Such properties only make sense when 
    # property is 
    # So that dropout and kept_components can be accessed as an atribute.
    @property
    def dropout_comp(self):
        alpha = torch.exp(self.log_alpha.detach())
        return alpha / (alpha + 1)

    @property
    def kept_components(self):
        #Get the ocmponents that were kept more.
        keep = (self.dropout_comp.reshape(-1) < self.dropout_threshold).tolist()
        components = [i for i, kept in enumerate(keep) if kept]
        return components

    def recon_loss(self, fwd_return, target=None, mask=None):
        """ 
        Reconstruction loss.

        Done to obtain mae and reconstruction, and also 
        from a target
        This function is not supposed to be used in the interior
        """

        X = fwd_return['x'] if target is None else target
        pxz = fwd_return['pxz']
        zp = fwd_return['zp']

        if mask is None:
            mask = [torch.ones(x.shape, dtype=torch.bool).to(self.device) for x in X]

        rec_loss = 0
        mae_loss = 0

        curr_channels = list(range(self.n_channels))
        t = 0
        #Mentre hi hagi canals uqe encara tinguin tps
        while len(curr_channels) > 0:
            #Select only channels that have that tp in their data
            curr_channels = [i for i, x_ch in enumerate(X) if t < x_ch.shape[0]]
            x = [x_ch[t, :, :] for i, x_ch in enumerate(X) if i in curr_channels]
            mask_ch = [mask_ch[t, :, 0] for i, mask_ch in enumerate(mask) if i in curr_channels]

            for i in range(len(curr_channels)):
                ch = curr_channels[i]
                # KL divergence
                #the second distribution is not the normal, is the prior!!

                for j in range(len(curr_channels)):
                    ch2 = curr_channels[j]
                    # i = latent comp; j = decoder
                    # Direct (i=j) and Crossed (i!=j) Log-Likelihood
                    # we only count the mae and reconstructions that exist!!
                    combined_mask = torch.logical_and(mask_ch[i], mask_ch[j])
                    if not torch.sum(combined_mask) == 0:
                        rec_loss += reconstruction_error(target=x[j], predicted=pxz[ch][t][ch2].loc, mask=combined_mask)
                        mae_loss += mae(target=x[j], predicted=pxz[ch][t][ch2].loc, mask=combined_mask)
            t += 1

        losses = {
            'rec_loss': rec_loss.item(),
            'mae': mae_loss.item()
        }

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
        self.kl_loss = {}
        for ch in range(self.n_channels):
            self.kl_loss[ch] = []

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