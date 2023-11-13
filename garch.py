import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import norm
import dataTransformations as dF
import arch
import numpy.random as nprand

#Get data and difference data
df=pd.read_csv("Data1997-2022.csv")

cpi_diff=dF.differencing(df.iloc[:,1],12)
cov_1=dF.differencing(df.iloc[:,2],12)
cov_2=dF.differencing(df.iloc[:,3],12)
cov_3=dF.differencing(df.iloc[:,4],12)
covs=np.dstack((cov_1,cov_2,cov_3)).squeeze()
r = np.zeros(len(cpi_diff))

#Calculate the returns from the response
for i in range(1,len(cpi_diff)):
    r[i] = (cpi_diff[i]-cpi_diff[i-1])/cpi_diff[i-1]

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
    for i in range(p):
        a+=params[i]*r_prev[-i]**2
    for i in range(q):
        b+=params[p+i]*sigma_prev[-i]**2
    # for i in range(m):
    #     g+=params[p+q+i]*covs[i]
    g=params[p+q+i:]@covs
    sum=a+b+g
    return np.sqrt(sum)


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
    r_new = [r[i] for i in range(len(r)-1,-1,-1)]
    sigma_new = [sigma[i] for i in range(len(sigma)-1,-1,-1)]
    r_i = np.append(r_new,1)
    r_i=np.array(r_i)
    sigma_new=np.array(sigma_new)
    K = np.concatenate((r_i[:n_a],sigma_new[:n_b],covs)) #Combine r_i and sigma into one vector. 
    L=1
    for ti in range(t):
        L=L*(1/(np.sqrt(2*np.pi*np.transpose(P)@K))*np.exp(-1/2*r[ti]**2/(np.transpose(P)@K)**2))
    l=-np.log(L)
    return l

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
    r_new = [r[i] for i in range(len(r)-1,-1,-1)]
    sigma_new = [sigma[i] for i in range(len(sigma)-1,-1,-1)]
    r_i = np.append(r_new,1)
    K = np.concatenate((r_i[:n_a], sigma_new[:n_b],covs)) #Combine r_i and sigma into one vector. 
    b=1/(np.transpose(P)@K)
    score = (len(r))/2 * (1/b) * np.transpose(K) -1/2 * sum(np.square(r))*(b)**(-2) * np.transpose(K)
    return score

def con_pos(x):
    "Constraint for the minimizer specifying positive parameters. "
    return np.abs(x)

def garch_fit(alphas_init,betas_init,tol,r,covs,maxiter,p,q,m,sigma_init,gammas_init,N):
    """ Function for fitting a GARCHX(p,q) model and estimating parameters.

    Paramaters:
    alphas_init: A ((p+1)x1) vector with initial guesses for the alphas
    betas_init: A (qx1) vector with initial guesses for the betas
    tol: Tolerance for allowed error, int.
    r: A (Nx1) vector with values for the returns
    covs: A (Nx3) vector with the data for the covariates
    maxiter: The maximum number of allowed iterations, int.
    p: The number of return regressors
    q: The number of sigma regressors
    m: the number of covariates
    sigma_init: Initial value for sigma, int
    gammas_init: A (3x1) vector with initial values for the gammas
    N: Total number of returns, int

    Fits a GARCH model with m covariates and estimates parameters for alphas, betas and gammas by using MLE.
    The function terminates when the change in variables are less than the toleranse.

    Returns: The estimated parameters, number of iterations, the returns used and the estimated sigmas.
    """
    not_tol=True
    i=0
    sigma_prevs=np.array([sigma_init])
    r_prevs=np.array([r[0]])
    n_a=len(alphas_init)
    n_b=len(betas_init)
    params_old=np.concatenate((alphas_init,betas_init,gammas_init))

    cons={"type":"eq","fun":con_pos}
    while not_tol and i<maxiter:
        c=covs[i,:]
        sigma=sigma_t_l(r_prevs,sigma_prevs,params_old,p,q,m,c)
        #minimize log likelihood and get parameter estimates
        if((len(params_old)-4)>(len(r_prevs)+len(sigma_prevs))):
            m=len(params_old)-3-(len(r_prevs)+len(sigma_prevs))
            m=int(m/2)
            b=int((len(params_old)-1)/2)
            b=int(b-m)
            params_old_s=np.concatenate((params_old[0:b+1],params_old[b+m+1:-(m+3)],params_old[-3:]))
            result=minimize(logLikelihood,params_old_s,method="SLSQP", jac = score_1,args=(r_prevs,sigma_prevs,c,i+1,n_a,n_b),constraints=cons)
        else:
            result=minimize(logLikelihood,params_old,method="SLSQP", jac = score_1,args=(r_prevs,sigma_prevs,c,i+1,n_a,n_b),constraints=cons)
        new_params=result.x
    

    #Check if difference in Euclidian norm are smaller than tol 
        if((len(params_old)-4)>(len(r_prevs)+len(sigma_prevs))):
            m=len(params_old)-3-(len(r_prevs)+len(sigma_prevs))
            m=int(m/2)
            b=int((len(params_old)-1)/2)
            b=int(b-m)
            params_old_s=np.concatenate((params_old[0:b+1],params_old[b+m+1:-(m+3)],params_old[-3:]))
            if np.linalg.norm(params_old_s - new_params)<tol:
                not_tol=False
            params_old[0:b+1]=new_params[0:b+1]
            params_old[b+m+1:-m]=new_params[-m:]
        else:
            if np.linalg.norm(params_old - new_params)<tol:
                not_tol=False
            params_old=new_params


        i+=1
        r_prevs=np.append(r_prevs,r[i])
        sigma_prevs=np.append(sigma_prevs,sigma)
    return params_old,i,sigma_prevs,r_prevs

p=2
q=2
params,i,sigma_prevs,r_prevs=garch_fit(alphas_init=np.array([0.005,0.005,0.005]),betas_init=np.array([0.005,0.005]),tol=1e-7,r=r,covs=covs,maxiter=100,p=2,q=2,m=3,sigma_init=0.05,gammas_init=np.array([0.01,0.01,0.01]),N=len(r))
print("params=",params)
print("i",i)
print("r_prevs",r_prevs)
print("sigma_prevs",sigma_prevs)

# model=arch.arch_model(cpi_diff,x=covs,mean="ARX",vol="GARCH",p=3,q=2)
# results=model.fit()
# print(results)

def predict_garch(params, r_prev, sig_prev, covs_prev, M, npred, p, q):
    r_pred=np.copy(r_prev)
    print(r_pred)
    for i in range(npred):
        yeet_r = np.flip(r_pred)
        yeet_sig = np.flip(sig_prev)
        sigma=sigma_t_l(r_prevs,sigma_prevs,params,p,q,3,covs[-1,:])
        mu=0
        preds = nprand.normal(mu,sigma,M)
        mp=np.mean(preds)
        r_pred = np.append(r_pred,mp)
    return r_pred

preds=predict_garch(params, r_prevs, sigma_prevs, covs, 100, 5, p, q)
print(preds)