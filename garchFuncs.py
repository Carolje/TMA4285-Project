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
    #r_new = [r[i] for i in range(len(r)-1,-1,-1)]
    #sigma_new = [sigma[i] for i in range(len(sigma)-1,-1,-1)]
    r_i = np.append(r_new,1)
    #New stuff
    iter_start = max(n_a, n_b)
    score = 0
    for ti in range(iter_start+1, t+1):
        K = np.concatenate((np.concatenate((r_i[0:1], r_i[-ti:-(ti-n_a)])), sigma_new[-ti:-(ti-n_b)], covs[ti,:]))
        b = b=1/(np.transpose(P)@K)
        score += (1/b) * np.transpose(K) -1/2 * sum(np.square(r))*(b)**(-2) * np.transpose(K)
    #K = np.concatenate((r_i[:n_a], sigma_new[:n_b],covs)) #Combine r_i and sigma into one vector. 
    #b=1/(np.transpose(P)@K)
    #score = (len(r))/2 * (1/b) * np.transpose(K) -1/2 * sum(np.square(r))*(b)**(-2) * np.transpose(K)
    return score

def con_pos(x):
    "Constraint for the minimizer specifying positive parameters. "
    return np.abs(x)

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
    #r_new = [r[i] for i in range(len(r)-1,-1,-1)]
    #sigma_new = [sigma[i] for i in range(len(sigma)-1,-1,-1)]
    r_i = np.append(r,1)
    r_i=np.array(r_i)
    r_new=np.flip(r_i)
    #K = np.concatenate((r_i[:n_a],sigma_new[:n_b],covs)) #Combine r_i and sigma into one vector.
    iter_start = max(n_a, n_b)
    l=0
    for ti in range(iter_start+1, t+1):
        r_temp=np.concatenate((r_new[0:1],r_new[-ti:-(ti-n_a)]))
        K = np.concatenate((r_temp, sigma_new[-ti:-(ti-n_b)], covs[ti,:]))
        l += - np.log(np.sqrt(2*np.pi)) - np.log(np.transpose(P)@K) -1/2*r_new[ti]**2/((np.transpose(P)@K))**2
        #L=L*(1/(np.sqrt(2*np.pi*np.transpose(P)@K))*np.exp(-1/2*r[ti]**2/(np.transpose(P)@K)**2))
    #l=-np.log(L)
    return float(l)


def summary(params,i,sigma_prevs,r_prevs,p,q,d,covs):
    """
    Prints the results 
    """
    print('{:^80}'.format("Results"))
    print("="*80)
    print('{:<25}'.format("Dep. Variable:"),'{:>25}'.format("CPI returns"))
    print('{:<25}'.format("Model:"),'{:>13}'.format("GARCH("),p,",",q,")")
    print('{:<25}'.format("No. Observations:"),'{:>25}'.format(len(r_prevs)))
    print('{:<25}'.format("AIC"),'{:>25}'.format("AICvalue"))
    print('{:<25}'.format("BIC"),'{:>25}'.format("BICvalue"))
    print('{:<25}'.format("Log Likelihood"),'{:>25}'.format("Logvalue"))
    print("="*80)
    print('{:>25}'.format("coef"),'{:>10}'.format("std.err"),'{:>10}'.format("z"),
            '{:>10}'.format("P>|z|"),'{:>10}'.format("[0.025"),'{:>10}'.format("0.975]"))
    print("-"*80)
    # differencing term
    if d == 0:
        print('{:<14}'.format("const"),'{:>10.4f}'.format(4.23589),
                '{:>10.4f}'.format(4.23589),'{:>10.4f}'.format(4.23589),
                '{:>10.4f}'.format(4.23589),'{:>10.4f}'.format(4.23589),
                '{:>10.4f}'.format(4.23589))
    # exogenious terms
    for (j,x) in enumerate(covs):
        print('{:<14}'.format(x),'{:>10.3e}'.format(covs[j]),
                '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589))
    # ar terms
    for j in range(p):
        print('{:<14}'.format("ar.L"+str(j+1)),'{:>10.3e}'.format(sigma_prevs),
                '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589))
    # ma terms
    for j in range(q):
        print('{:<14}'.format("ma.L"+str(j+1)),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                '{:>10.3e}'.format(4.23589))
    print("="*80)