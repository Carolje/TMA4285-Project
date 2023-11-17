import numpy as np
import pandas as pd
from scipy.optimize import minimize, approx_fprime
from numpy.linalg import norm
import dataTransformations as dF
import arch
import numpy.random as nprand
import garchFuncs as gF
import matplotlib.pyplot as plt

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

def garch_fit(alphas_init,betas_init,tol,r,covs,maxiter,p,q,m,sigma_init,gammas_init,N):
    """ Function for fitting a GARCHX(p,q) model and estimating parameters.

    Paramaters:
    alphas_init: A ((p+1)x1) vector with initial guesses for the alphas
    betas_init: A (qx1) vector with initial guesses for the betas
    tol: Tolerance for allowed error for the parameters, int.
    r: A (Nx1) vector with values for the returns
    covs: A (Nx3) vector with the data for the covariates
    maxiter: The maximum number of allowed iterations, int.
    p: The number of return regressors
    q: The number of sigma regressors
    m: the number of covariates
    sigma_init: Initial value for sigma, int
    gammas_init: A (3x1) vector with initial values for the gammas
    N: Total number of returns, int

    Fits a GARCH model with m covariates and estimates parameters for alphas, betas and gammas by minimizing the 
    negative log-likelihood. Note that we have set a quite high toleranse in the minimize function due to the flat 
    log-likelihood.
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
        print("iteration",i)
        c=covs[i+3,:]
        sigma=gF.sigma_t_l(r_prevs,sigma_prevs,params_old,p,q,m,c)        
        result=minimize(gF.logLikelihood,params_old,method="SLSQP",args=(r_prevs,sigma_prevs,covs[:i+3,:],i,n_a-1,n_b),constraints=cons,tol=1e-2, options = {"maxiter": 20000,
                                                                                                                                                              "disp": False}) #,jac=gF.score_1
        new_params=result.x
    
        if np.linalg.norm(params_old - new_params)<tol:
            not_tol=False
        params_old=new_params
        i+=1
        r_prevs=np.append(r_prevs,r[i])
        sigma_prevs=np.append(sigma_prevs,sigma)
    return params_old,i,sigma_prevs,r_prevs

"""Fitting different GARCH models to see which model is best"""

# GARCH(2,2)
p=2
q=2
a=np.array([0.5,0.5,0.5])
b=np.array([0.2,0.2])
g=np.array([0.1,0.2,0.3])
sigma_init = np.sqrt(gF.sample_variance(r))
params,i,sigma_prevs,r_prevs=garch_fit(alphas_init=a,betas_init=b, tol=1e-7,r=r,covs=covs,maxiter=100,p=p,q=q,m=3,sigma_init=sigma_init,gammas_init=g,N=len(r))
sigma_1=np.array([sigma_init])
n_a = p
n_b = q
for j in range(len(r)-1):
        sigma=gF.sigma_t_l(r[:j+1],sigma_1,params,p,q,3,covs[j,:])
        sigma_1=np.append(sigma_1,sigma)
AIC_1=gF.AIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)
BIC_1=gF.BIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)

#GARCH(1,1)
p = 1
q = 1
a = np.array([0.5,0.5])
b = np.array([0.2])
params,i,sigma_prevs,r_prevs=garch_fit(alphas_init=a,betas_init=b, tol=1e-7,r=r,covs=covs,maxiter=100,p=p,q=q,m=3,sigma_init=sigma_init,gammas_init=g,N=len(r))
sigma_1=np.array([sigma_init])
n_a = p
n_b = q
for j in range(len(r)-1):
        sigma=gF.sigma_t_l(r[:j+1],sigma_1,params,p,q,3,covs[j,:])
        sigma_1=np.append(sigma_1,sigma)
AIC_2=gF.AIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)
BIC_2=gF.BIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)

#GARCH(3,4)
p = 3
q = 4
a = np.array([0.5,0.5,0.5,0.5])
b = np.array([0.2,0.2,0.2,0.2])
params,i,sigma_prevs,r_prevs=garch_fit(alphas_init=a,betas_init=b, tol=1e-7,r=r,covs=covs,maxiter=100,p=p,q=q,m=3,sigma_init=sigma_init,gammas_init=g,N=len(r))
sigma_1=np.array([sigma_init])
n_a = p
n_b = q
for j in range(len(r)-1):
        sigma=gF.sigma_t_l(r[:j+1],sigma_1,params,p,q,3,covs[j,:])
        sigma_1=np.append(sigma_1,sigma)
AIC_3=gF.AIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)
BIC_3=gF.BIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)

#GARCH(2,3)
p = 2
q = 3
a = np.array([0.5,0.5,0.5])
b = np.array([0.2, 0.2, 0.2])
params,i,sigma_prevs,r_prevs=garch_fit(alphas_init=a,betas_init=b, tol=1e-7,r=r,covs=covs,maxiter=100,p=p,q=q,m=3,sigma_init=sigma_init,gammas_init=g,N=len(r))
sigma_1=np.array([sigma_init])
n_a = p
n_b = q
for j in range(len(r)-1):
        sigma=gF.sigma_t_l(r[:j+1],sigma_1,params,p,q,3,covs[j,:])
        sigma_1=np.append(sigma_1,sigma)
AIC_4=gF.AIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)
BIC_4=gF.BIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)

#GARCH(3,3)
p = 3
q = 3
a = np.array([0.5,0.5,0.5,0.5])
b = np.array([0.2,0.2,0.2])
params,i,sigma_prevs,r_prevs=garch_fit(alphas_init=a,betas_init=b, tol=1e-7,r=r,covs=covs,maxiter=100,p=p,q=q,m=3,sigma_init=sigma_init,gammas_init=g,N=len(r))
sigma_1=np.array([sigma_init])
n_a = p
n_b = q
for j in range(len(r)-1):
        sigma=gF.sigma_t_l(r[:j+1],sigma_1,params,p,q,3,covs[j,:])
        sigma_1=np.append(sigma_1,sigma)
AIC_5=gF.AIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)
BIC_5=gF.BIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)

#Vector with AIC and BIC values for all the testet GARCH models
vec_AIC = np.array([AIC_1, AIC_2, AIC_3, AIC_4, AIC_5]) 
vec_BIC = np.array([BIC_1, BIC_2, BIC_3, BIC_4, BIC_5])
#Indices for the model with best AIC and BIC
min_AIC = np.argmin(vec_AIC)
min_BIC = np.argmin(vec_BIC)

def get_jacobian(params,*args):
    """Function to find the gradient of the log-likelihood"""
    return approx_fprime(params, gF.logLikelihood,1.4901161193847656e-08,r,sigma_1,covs[:len(r),:],len(r)-1,n_a,n_b)


def get_hessian(params):
    """Function to find the hessian from the gradient of the log-likelihood"""
    return approx_fprime(params, get_jacobian,1.4901161193847656e-08,r,sigma_1,covs[:len(r),:],len(r)-1,n_a,n_b)

#The hessian for the log-likelihood calculated with the params for the GARCH(3,3) model which was the best model.
h=get_hessian(params)

#Calculating CIs for the best models
conf_ints = gF.CIs(h)

#Get a summary print out of estimates and diagnostics for the GARCH(3,3) model which was the best model
met=[vec_AIC[min_AIC],vec_BIC[min_BIC]]
gF.summary(params, r, p, q, met, conf_ints)

#Predictions for the GARCH(3,3) and plot of the predictions
n_pred=50
r_preds,x_preds=gF.predict_garch(params, r,cpi_diff, sigma_1,covs, 1000, n_pred, 3, 3)
t=np.linspace(300,300+n_pred,n_pred)
plt.figure()
plt.plot(x_preds[:-n_pred],label="Differenced CPI")
plt.plot(t,x_preds[-n_pred:],label="Prediction")
plt.xlabel("Months after January 1997")
plt.ylabel("Differenced CPI")
plt.legend()
plt.show()


#Pythons arch implementataion
# model=arch.arch_model(cpi_diff,x=covs,mean="ARX",vol="GARCH",p=4,q=3)
# results=model.fit()
# print(results)