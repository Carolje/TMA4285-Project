import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import norm
import dataTransformations as dF
#import arch
import numpy.random as nprand
import garchFuncs as gF

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

def sample_variance(vec):
    return np.sum((vec-np.mean(vec))**2)/(len(vec)-1)


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
    sigma_prevs=np.array([sigma_init])
    r_prevs=r[0:p+2]
    params_old=np.concatenate((alphas_init,betas_init,gammas_init))
    for i in range(len(r_prevs)-1):
        sigma=gF.sigma_t_l(r_prevs[:i+1],sigma_prevs,params_old,p,q,m,covs[i,:])
        sigma_prevs=np.append(sigma_prevs,sigma)
    n_a=len(alphas_init)
    n_b=len(betas_init)
    cons={"type":"eq","fun":gF.con_pos}
    i=max(p,q)+1
    while not_tol and i<maxiter:
        print(i)
        c=covs[i+3,:]
        sigma=gF.sigma_t_l(r_prevs,sigma_prevs,params_old,p,q,m,c)        
        result=minimize(gF.logLikelihood,params_old,method="SLSQP",args=(r_prevs,sigma_prevs,covs[:i+3,:],i,n_a-1,n_b),constraints=cons,tol=1e-12, options = {"maxiter": 20000,
                                                                                                                                                              "disp": True}) #,jac=gF.score_1
        new_params=result.x
    
        if np.linalg.norm(params_old - new_params)<tol:
            not_tol=False
        params_old=new_params
        i+=1
        r_prevs=np.append(r_prevs,r[i])
        sigma_prevs=np.append(sigma_prevs,sigma)
    return params_old,i,sigma_prevs,r_prevs

p=2
q=2
a=np.array([0.5,0.2,0.3])
b=np.array([0.1,0.2])
g=np.array([0.1,0.2,0.3])
sigma_init = np.sqrt(sample_variance(r))
params,i,sigma_prevs,r_prevs=garch_fit(alphas_init=a,betas_init=b, tol=1e-7,r=r,covs=covs,maxiter=100,p=p,q=q,m=3,sigma_init=sigma_init,gammas_init=g,N=len(r))
print("params=",params)
#print("i",i)
#print("r_prevs",r_prevs)
#print("sigma_prevs",sigma_prevs)

#print(gF.AIC(len(params),params,r_prevs,sigma_prevs,covs[i,:],t,n_a,n_b))
# model=arch.arch_model(cpi_diff,x=covs,mean="ARX",vol="GARCH",p=3,q=2)
# results=model.fit()
# print(results)

def predict_garch(params, r_prev, sig_prev, covs_prev, M, npred, p, q):
    r_pred=np.copy(r_prev)
    for i in range(npred):
        sigma=gF.sigma_t_l(r_prevs,sigma_prevs,params,p,q,3,covs[-1,:])
        mu=0
        preds = nprand.normal(mu,sigma,M)
        mp=np.mean(preds)
        r_pred = np.append(r_pred,mp)
    return r_pred

#preds=predict_garch(params, r_prevs, sigma_prevs, covs, 100, 5, p, q)

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

r_test, r_train, covs_test, covs_train = train_test_split(r, covs, 0.7)
#print("r_test",len(r_test))
#print("r_train",len(r_train))
#print("covs_test",np.shape(covs_test))
#print("covs_train",np.shape(covs_train))


def AIC(k,P,r_prevs,sigma_prevs,covs,t,n_a,n_b):
    AIC=2*k+2*gF.logLikelihood(P,r_prevs,sigma_prevs,covs,t,n_a,n_b)
    return AIC

def BIC(k,P,r_prevs,sigma_prevs,covs,t,n_a,n_b):
    BIC=k*np.log(len(r_prevs))+2*gF.logLikelihood(P,r_prevs,sigma_prevs,covs,t,n_a,n_b)
    return BIC

sigma_prevs=np.array([sigma_init])
n_a = p
n_b = q
for i in range(len(r)-1):
        sigma=gF.sigma_t_l(r[:i+1],sigma_prevs,params,p,q,3,covs[i,:])
        sigma_prevs=np.append(sigma_prevs,sigma)

print("AIC",AIC(len(params),params,r,sigma_prevs,covs,len(r)-1,n_a,n_b))
print("BIC",BIC(len(params),params,r,sigma_prevs,covs,len(r)-1,n_a,n_b))
