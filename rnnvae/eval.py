"""
Functions to evaluate the results of the network.

Mainly, functions for prediction and reconstruction.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./'))
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch import nn

def my_mean_absolute_error(y_true, y_pred):
    """
    Auxiliar function to compute the mean and std of the absolute error
    """

    maes_list = [mean_absolute_error(y_t, y_p) for (y_t, y_p) in zip(y_true, y_pred)]

    mean_mae = np.mean(maes_list)
    std_mae = np.std(maes_list)
    return mean_mae, std_mae



def eval_prediction(model, X_test, t_pred, pred_ch, DEVICE):
    """ Evaluate prediction of a set of time points for a trained model.
    We specify the number of timepoints tp of the data to predict.

    Then, we predict using only the last T-tp timepoints.
    
    Finally, we compute the MAE for each channel in pred_ch

    model: trained model of RNNVAE
    X_test: data, not processed to go into the model
    t_pred: number of timepoints to predict
    pred_ch: channels we have to predict
    DEVICE: device where the model is
    Returns: list of MAE of the prediction for each channel in pred_ch
    """
    assert model.is_fitted, "Model is not fitted!"

    #process the data
    X_test_minus = []
    mask_test_minus = []
    for x_ch in X_test:
        #Want to select only last T-t_pred values
        X_test_tensor = [ torch.FloatTensor(t[:-t_pred,:]) for t in x_ch]
        X_test_tensor_full = [ torch.FloatTensor(t) for t in x_ch]
        X_test_pad = nn.utils.rnn.pad_sequence(X_test_tensor, batch_first=False, padding_value=np.nan)
        mask_test = ~torch.isnan(X_test_pad)
        mask_test_minus.append(mask_test.to(DEVICE))
        X_test_pad[torch.isnan(X_test_pad)] = 0
        X_test_minus.append(X_test_pad.to(DEVICE))

    #Number of time points is the length of the input channel we want to predict
    ntp = max(np.max([[len(xi) for xi in x] for x in X_test]), np.max([[len(xi) for xi in x] for x in X_test]))
    # Run prediction
    X_test_fwd_minus = model.predict(X_test_minus, mask_test_minus, nt=ntp)
    X_test_xnext = X_test_fwd_minus["xnext"]

    #WAIT, això funciona amb prediccions només a l'ultim timepoint, o seleccionem a tots?
    #TODO: ADAPT FOR ANY NUMBER OF VALUES
    #THIS ONLY WORKS WITH T=1, NOT WITH THE OTHERS
    # Test data without last timepoint
    # X_test_tensors do have the last timepoint
    results= []
    for (X_ch, i) in zip([x for (i,x) in enumerate(X_test) if i in pred_ch], pred_ch):

        y_true = []
        #create y_true
        for x in X_ch:
            if len(x) > t_pred:
                for k in reversed(range(t_pred)):
                    # this is on the opposite order so that it works for t_pred > 1
                    # because if not, comparison doesnt work
                    y_true.append(x[-k])

        last_tp = [[len(x)-(tx+1) for tx in range(t_pred)] for x in X_ch] # last tp is max size of original data minus t_pred
        y_pred = []
        # for each subject, select the last tps (that we want to predict) NOT ONLY ONE
        j = 0
        for tp in last_tp:
            if tp[0] < t_pred:
                j += 1
                continue # ignore tps with only baseline
            for tpx in tp:
                y_pred.append(X_test_xnext[i][tpx, j, :])
            j += 1
        #Process it to predict it
        err, err_std = my_mean_absolute_error(y_true, y_pred)
        err = str(np.round(err, 2)) + '\pm' + str(np.round(err_std, 2))
        results.append(err)

        # mae_tp_ch = mean_absolute_error(y_true, y_pred)
        # results.append(mae_tp_ch)
    return results


def eval_reconstruction(model, X, X_test, mask_test, av_ch, recon_ch):
    """ Evaluate reconstruction of a list of channels given another list of channels, from a trained model
    model: trained model of RNNVAE
    X: data not being processed
    X_test: data already processed
    mask_test_list: mask of the data, already processed
    av_ch: list of channels to use (indexes)
    recon_ch: single index of channel to reconstruct

    Returns: MAE of the reconstruction over the test set
    """
    assert model.is_fitted, "Model is not fitted!"
    
    #Number of time points is the length of the channel we want to reconstruct
    ntp = len(X_test[recon_ch])
    #ntp = 1

    # try to reconstruct it from the other ones
    ch_recon = model.predict(X_test, mask_test, nt=ntp, av_ch=av_ch, task='recon')

    #Compare it to the channel we want to reconstruct
    y_true = X[recon_ch]

    # swap dims to iterate over subjects
    #selecting only the real timepoints
    y_pred = np.transpose(ch_recon["xnext"][recon_ch], (1,0,2)) # transpose to obtain dim that are comparable to X_test
    y_pred = [x_pred[:len(x_true)] for (x_pred, x_true) in zip(y_pred, y_true)]

    #prepare it timepoint wise
    #TODO: DO WE REALLY NEED TO DO THIS? SHOULDNT WE TREAT EACH SUBJECT JOINTLY?
    #y_pred = [subj for subj in y_pred]
    #y_true = [subj for subj in y_true]
    y_pred = [tp for subj in y_pred for tp in subj]
    y_true = [tp for subj in y_true for tp in subj]
    # y_pred = [np.random.rand(*x.shape) for x in y_pred]

    err, err_std = my_mean_absolute_error(y_true, y_pred)
    err = str(np.round(err, 2)) + '\pm' + str(np.round(err_std, 2))
    return err

    # mae_rec = mean_absolute_error(y_true, y_pred)
    # return mae_rec
