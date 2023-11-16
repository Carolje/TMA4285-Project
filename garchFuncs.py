import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import norm
import numpy.random as nprand

def sample_variance(vec):
    return np.sum((vec-np.mean(vec))**2)/(len(vec)-1)

def r_t(sigma,epsilon):
    "Function to calculate the r_t from the previous sigma."
    return sigma*epsilon

def sigma_t_l(r_prev,sigma_prev,params,p,q,m,covs): 
    """Function to calculate sigma from the previous r's, sigma's, covariates from the current timesteps and estimated params.
    
    Parameters:
    r_prev: the returns from the current and previous timesteps
    sigma_prevs: the sigma estimates for previous time steps
    params: estimated paramameters for alphas, betas and gammas
    p: The number of return regressors
    q: The number of sigma regressors
    m: the number of covariates
    covs: matrix of covariates for the current timestep

    returns: The new calculated sigma value. 
    """
    
    r_prev = np.append(r_prev,1)
    a=0
    b=0
    g=0
    g=params[p+q+1:]@covs
    if p>len(r_prev):
        p=len(r_prev)
    for i in range(p):
        a+=params[i]*r_prev[-i]**2
    if q>len(sigma_prev):
        q=len(sigma_prev)
    for i in range(q):
        b+=params[p+i]*sigma_prev[-i]**2
    
    sum=a+b+g
    return np.sqrt(sum)

def score_1(P,*args):
    """
    Computes the score vector of the log-likelihood of a GARCH(q,p)-model with respect to the vector
    where vectors alphas and betas are concatenated.

    Parameters:
    alphas: A ((p+1)x1) vector.
    betas: A (qx1) vector.
    r: A (Nx1) vector.
    sigma: A (Nx1) vector.

    Returns: score, a ((p+q+1)x1) vector.
    """
    r,sigma,covs,t,n_a,n_b=args
    r_new = np.flip(r)
    sigma_new = np.flip(sigma)
    r_i = np.append(r_new,1)
    iter_start = max(n_a, n_b)
    score = 0
    for ti in range(iter_start+1, t+1):
        K = np.concatenate((np.concatenate((r_i[0:1], r_i[-ti:-(ti-n_a)])), sigma_new[-ti:-(ti-n_b)], covs[ti,:]))
        b = b=1/(np.transpose(P)@K)
        score += 1/2*(1/b) * np.transpose(K) -1/2 * sum(np.square(r))*(b)**(-2) * np.transpose(K)
    return score

def Hessian_1(P,*args):
    """
    Computes the Hessian matrix of the log-likelihood of a GARCH(q,p) with respect to the vector
    where vectors alphas and betas are concatenated.

    Parameters:
    alphas: A ((p+1)x1) vector.
    betas: A (qx1) vector.
    r: A (Nx1) vector.
    sigma: A (Nx1) vector.
    
    Returns: hess, a ((p+1+q+m)x(p+1+q+m)) matrix.
    """
    print(args)
    r,sigma,covs,t,n_a,n_b=args
    r_new = np.flip(r)
    sigma_new = np.flip(sigma)
    r_i = np.append(r_new,1)

    iter_start = max(n_a, n_b)
    hess = np.zeros((len(P),len(P)))
    for ti in range(iter_start+1, t+1):
        K = np.concatenate((np.concatenate((r_i[0:1], r_i[-ti:-(ti-n_a)])), sigma_new[-ti:-(ti-n_b)], covs[ti,:]))
        b = b=1/(np.transpose(P)@K)
        hess += -1/2*(1/b)**(-2) * K@np.transpose(K) +1/4 * sum(np.square(r))*(b)**(-3) * K@np.transpose(K)
    return hess

def CIs(hess):
    hess=np.linalg.inv(hess)
    d=np.diag(hess)
    conf_ints=np.zeros((len(d),2))
    for i in range(len(d)):
        u=1.96*d[i]
        l=-1.96*d[i]
        conf_ints[i,:]=[l,u]
    return conf_ints



def con_pos(x):
    "Constraint for the minimizer specifying positive parameters. "
    return x**2

def logLikelihood(P,*args):
    """ Function that calculates the log likelihood.

    Parameters:
    P: A (p+q+m)x1) vector with parameter estimates for alphas, betas and gammas. The parameters inn
        these vector gets estimated by MLE.
    *args:
    r: A (px1) vector of previous return values.
    sigma: A (q+1) vector with the sigma estimates for previous time steps
    covs: A (1xm) matrix of covariates for the current timestep
    t: The current timestep
    n_a: The number of alphas
    n_b: The number of betas

    return: the calculated log-likelihood
    

    """
    r,sigma,covs,t,n_a,n_b=args
    sigma_new = np.flip(sigma)
    r_i = np.append(r,1)
    r_i=np.array(r_i)
    r_new=np.flip(r_i)
    iter_start = max(n_a, n_b)
    l=0
    for ti in range(iter_start+1, t+1):
        r_temp=np.concatenate((r_new[0:1],r_new[-ti:-(ti-n_a)]))
        K = np.concatenate((r_temp, sigma_new[-ti:-(ti-n_b)], covs[ti,:]))
        if np.sqrt(2*np.pi)<0 or (np.transpose(P)@K)<0:
            print(np.sqrt(2*np.pi))
            print(np.transpose(P)@K)
            print(ti)
        l += -1/2*np.log(np.sqrt(2*np.pi)) -1/2*np.log(np.transpose(P)@K) -1/2*r_new[ti]**2/((np.transpose(P)@K))**2
    return float(-l)

def AIC(k,P,r_prevs,sigma_prevs,covs,t,n_a,n_b):
    AIC=2*k+2*logLikelihood(P,r_prevs,sigma_prevs,covs,t,n_a,n_b)
    return AIC

def BIC(k,P,r_prevs,sigma_prevs,covs,t,n_a,n_b):
    BIC=k*np.log(len(r_prevs))+2*logLikelihood(P,r_prevs,sigma_prevs,covs,t,n_a,n_b)
    return BIC
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
def AIC(k,P,r_prevs,sigma_prevs,covs,t,n_a,n_b):
    AIC=2*k-2*logLikelihood(P,r_prevs,sigma_prevs,covs,t,n_a,n_b)
    return AIC

def BIC(k,P,r_prevs,sigma_prevs,covs,t,n_a,n_b):
    BIC=k*np.log(len(r_prevs))-2*logLikelihood(P,r_prevs,sigma_prevs,covs,t,n_a,n_b)
    return BIC

def summary(params,r_prevs,p,q,m):
    """
    Prints the results 
    """
    print('{:^80}'.format("Results"))
    print("="*80)
    print('{:<25}'.format("Dep. Variable:"),'{:>25}'.format("CPI returns"))
    print('{:<25}'.format("Model:"),'{:>13}'.format("GARCH("),p,",",q,")")
    print('{:<25}'.format("No. Observations:"),'{:>25}'.format(len(r_prevs)))
    print('{:<25}'.format("AIC"),'{:>25}'.format(m[0]))
    print('{:<25}'.format("BIC"),'{:>25}'.format(m[1]))
    print('{:<25}'.format("Log Likelihood"),'{:>25}'.format(m[2]))
    print("="*80)
    print('{:>25}'.format("coef"),'{:>10}'.format("std.err"),'{:>10}'.format("z"),
            '{:>10}'.format("P>|z|"),'{:>10}'.format("[0.025"),'{:>10}'.format("0.975]"))
    print("-"*80)

    names=[]
    for i in range(p+1):
        names.append(f"alpha {i}")
    for i in range(q):
        names.append(f"beta {i+1}")
    names.append("Unemployed rate")
    names.append("Policy rate")
    names.append("Monthly salary")
    for j in range(len(params)):
        print('{:<14}'.format(names[j]),'{:>10.3e}'.format(params[j]),
                '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589))
    print("="*80)