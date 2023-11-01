import numpy as np
import pandas as pd
def r_t(rho,epsilon):
    return rho*epsilon
def rho_t(r_prev,rho_prev,alphas,betas,p,q):
    a=0
    b=0
    for i in range(p):
        a+=alphas[i+1]*r_prev[i]**2
    for i in range(q):
        b+=betas[i]*rho_prev[i]^2
    return np.sqrt(alphas[0] + a+b)

def garch_fit():
