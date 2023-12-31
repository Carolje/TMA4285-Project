import numpy.random as nprand
import numpy as np
import random as rand

def predict_garch(alphas, betas, gammas, r_prev, sig_prev, params_pred, N, npred, p, q):
    preds = np.ones(N)
    for i in range(npred):
        for j in range(N):
            yeet_r = np.flip(r_prev)
            yeet_sig = np.flip(sig_prev)
            preds[j] = nprand.normal(0, alphas[0] + np.transpose(alphas[1:])@yeet_r[:p] + np.transpose(betas)@yeet_sig[:q] + np.transpose(gammas)@params_pred[-1])
        r_prev = np.append(r_prev,np.mean(preds))
    return r_prev[-npred:]


def sample_variance(vec):
    return np.sum((vec-np.mean(vec))**2)/(len(vec)-1)

#p-values, plot code, test error. 

def train_test_split(response, covs, perc):
    n = len(response)
    index_train = np.array(range(0,n-1))[0:int((n-1)*perc)]
    mask = np.full(len(response),True,dtype=bool)
    mask[index_train] = False
    response_test = response[mask]
    response_train = response[~mask]
    covs_test = covs[mask,]
    covs_train = covs[~mask,]
    return (response_test, response_train, covs_test, covs_train)

#Test error. 