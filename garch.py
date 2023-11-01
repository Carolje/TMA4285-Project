import numpy as np
import pandas as pd
def r_t(sigma,epsilon):
    return sigma*epsilon
def sigma_t(r_prev,sigma_prev,alphas,betas,p,q):
    #Need to be checked
    a=0
    b=0
    for i in range(p):
        a+=alphas[i+1]*r_prev[i]**2
    for i in range(q):
        b+=betas[i]*sigma_prev[i]^2
    return np.sqrt(alphas[0] + a+b)
def likelihood(r,sigma,T):
    L=1
    for t in range(T):
        L=L*(1/(np.sqrt(2*np.pi)*sigma[t])*np.exp(-1/2*r[t]**2/sigma[t]**2))
    l=-np.log(L)
    return l
def garch_fit(alphas_init, betas_init,tol,rs,maxiter):
    sigma=sigma_t(r_prev,sigma_prev,alphas,betas,p,q)

    #minimize log likelihood and get parameter estimates

    #Calculate score vector and Hessian

    #Cehck if difference for all params alpha and beta are smaller than tol

    #If yes break loop, if not continue to iterate and smaller than maxiter


