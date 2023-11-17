import numpy as np
import pandas as pd
from scipy.optimize import minimize, approx_fprime
from numpy.linalg import norm
import dataTransformations as dF
#import arch
import numpy.random as nprand
import garchFuncs as gF
import matplotlib.pyplot as plt
#from autograd import hessian

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
#p=2
#q=2
#a=np.array([0.5,0.2,0.3])
#b=np.array([0.1,0.2])
#g=np.array([0.1,0.2,0.3])
#sigma_init = np.sqrt(gF.sample_variance(r))
#params,i,sigma_prevs,r_prevs=gF.garch_fit(alphas_init=a,betas_init=b, tol=1e-7,r=r,covs=covs,maxiter=100,p=p,q=q,m=3,sigma_init=sigma_init,gammas_init=g,N=len(r))

#sigma_1=np.array([sigma_init])
#n_a = p
#n_b = q
#for i in range(len(r)-1):
        #sigma=gF.sigma_t_l(r[:i+1],sigma_1,params,p,q,3,covs[i,:])
        #sigma_1=np.append(sigma_1,sigma)

#AIC=gF.AIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)
#BIC=gF.BIC(len(params),params,r,sigma_1,covs,len(r)-1,n_a,n_b)
#print(params,r_prevs,sigma_prevs,len(r_prevs)-1,n_a-1,n_b)

#a=r_prevs,sigma_prevs,covs[:len(r_prevs),:],len(r_prevs)-1,n_a,n_b

#h=gF.get_hessian(params)

#conf_ints=gF.CIs(h)
#print("c",conf_ints)
#m=[AIC,BIC]
#print(m)

#gF.summary(params,r_prevs,p,q,m,conf_ints)

#Grid of possible values for p and q:
k = 5
sigma_init = np.sqrt(gF.sample_variance(r))
g = np.array([0.1, 0.2, 0.3])
vector_AIC = np.zeros(k)
vector_BIC = np.zeros(k)
for i in range(1,k+1):
    a = np.repeat(0.5, i +1)
    b = np.repeat(0.2, i)
    print(a)
    print(b)
    params,s,sigma_prevs,r_prevs=gF.garch_fit(alphas_init=a,betas_init=b, tol=1e-7,r=r,covs=covs,maxiter=100,p=i,q=i,m=3,sigma_init=sigma_init,gammas_init=g,N=len(r))
    sigma_1=np.array([sigma_init])
    for d in range(len(r)-1):
        sigma=gF.sigma_t_l(r[:d+1],sigma_1,params,i,i,3,covs[d,:])
        sigma_1=np.append(sigma_1,sigma)
        vector_AIC[i]=gF.AIC(len(params),params,r,sigma_1,covs,len(r)-1,i,i)
        vector_BIC[i]=gF.BIC(len(params),params,r,sigma_1,covs,len(r)-1,i,i)

print("AIC matrix", vector_AIC)
print("BIC matrix", vector_BIC)