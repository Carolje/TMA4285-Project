import numpy.random as nprand
import numpy as np


def predict_garch(alphas, betas, gammas, r_prev, sig_prev, params_pred, N, npred, p, q):
    preds = np.ones(N)
    for i in range(npred):
        for j in range(N):
            yeet_r = np.flip(r_prev)
            yeet_sig = np.flip(sig_prev)
            preds[j] = nprand.normal(0, alphas[0] + np.transpose(alphas[1:])@yeet_r[:p] + np.transpose(betas)@yeet_sig[:q] + np.transpose(gammas)@params_pred[-1])
        r_prev = np.append(r_prev,np.mean(preds))
    return r_prev[-npred:]