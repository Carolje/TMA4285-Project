def score(alphas,betas,r_prevs,r_t,sigma_prevs,sigma_t,N):
    #Are supposed to be a sum over T in b, but that are taken care of by the while loop? 
    # Have removed it because I was unsure of the sum. Need to fix that. 
    n=len(alphas)+len(betas)
    vec=np.zeros(n)
    b=1/(np.transpose(alphas)@r_prevs+np.transpose(betas)@sigma_prevs)
    vec[0]=N/2*b+1/2*np.sum(r_prevs**2)*b
    for i in range(1,len(alphas)):
        vec[i]=N/2*b*r_prevs[i-1]**2+1/2*r_t**2*r_prevs[i-1]**2*b
    for i in range(len(betas)):
        vec[len(alphas)+i]=N/2*b*sigma_prevs[i]**2+1/2*sigma_t**2*sigma_prevs[i]**2*b
    return vec

def Hessian(alphas,betas,r_prevs,r_t,sigma_prevs,sigma_t,N):
    
    n_a=len(alphas)
    n_b=len(betas)
    n=n_a+n_b
    vec=np.zeros((n,n))
    b=1/(np.transpose(alphas)@r_prevs+np.transpose(betas)@sigma_prevs)
    vec[0,0]=-N/2*b**2-1/2*r_t**2*b**2
    for i in range(1,n_a):
        vec[0,i]=-N/2*b**2*r_prevs[i-1]**2-1/2*r_t**2*r_prevs[i-1]**2*b**2
        vec[i,0]=-N/2*b**2*r_prevs[i-1]**2-1/2*r_t**2*r_prevs[i-1]**2*b**2
    for i in range(n_b):
        vec[0,n_a+i]=-N/2*b**2*sigma_prevs[i]**2-1/2*r_t**2*sigma_prevs[i]**2*b**2
        vec[n_a+i,0]=-N/2*b**2*sigma_prevs[i]**2-1/2*r_t**2*sigma_prevs[i]**2*b**2
    for i in range(1,n_a):
        for j in range(1,n_a):
            vec[i,j]=-N/2*b**2*r_prevs[i-1]**2*r_prevs[j-1]**2-1/2*r_t**2*r_prevs[i-1]**2*r_prevs[j-1]**2*b**2
    for i in range(1,n_a):
        for j in range(n_a,n):
            vec[i,j]=-N/2*b**2*r_prevs[i-1]**2*sigma_prevs[j-1]**2-1/2*r_t**2*r_prevs[i-1]**2*sigma_prevs[j-1]**2*b**2
    for i in range(n_a,n):
        for j in range(n_a,n):
            vec[i,j]=-N/2*b**2*sigma_prevs[i-1]**2*sigma_prevs[j-1]**2-1/2*r_t**2*sigma_prevs[i-1]**2*sigma_prevs[j-1]**2*b**2
    for i in range(n_a,n):
        for j in range(1,n_a):
            vec[i,j]=-N/2*b**2*sigma_prevs[i-1]**2*r_prevs[j-1]**2-1/2*r_t**2*sigma_prevs[i-1]**2*r_prevs[j-1]**2*b**2
    return vec

def likelihood(r,sigma,T):
    L=1
    for t in range(T):
        L=L*(1/(np.sqrt(2*np.pi)*sigma[t])*np.exp(-1/2*r[t]**2/sigma[t]**2))
    l=-np.log(L)
    return l
#What is N?

def garch_fit(alphas_init, betas_init,tol,r,maxiter,p,q,sigma_init,N):
    not_tol=True
    i=0
    sigma_prevs=[]
    r_prevs=[r[0]]
    n_a=len(alphas_init)
    n_b=len(betas_init)
    params_old=[alphas_init,betas_init]
    while not_tol and i<maxiter:
        sigma=sigma_t(r_prevs,sigma_prevs,params_old[:n_a],params_old[n_a:],p,q)
        
        #minimize log likelihood and get parameter estimates
        result=minimize(likelihood,params_old,method="BFGS", jac = score_1, hess = Hessian_1)
        new_params=result.x
    

    #Calculate score vector and Hessian
        alphas=new_params[:n_a]
        betas=new_params[:n_b]
        s=score(alphas,betas,r_prevs,r[i+1],sigma_prevs,sigma,N)
        H=Hessian(alphas,betas,r_prevs,r[i+1],sigma_prevs,sigma,N)

    #Update params
        params_new=params_old-np.linalg.inv(H)@s
    #Check if difference for all params alpha and beta are smaller than tol
        if norm(new_params-params_old)<tol:
            not_tol=False

        i+=1
        params_old=params_new
    #If yes break loop, if not continue to iterate and smaller than maxiter

def sigma_t(r_prev,sigma_prev,alphas,betas,p,q):
    r_prev = np.append(r_prev,1)
    a=0
    b=0
    for i in range(p):
        a+=alphas[i]*r_prev[-i]**2
    for i in range(q):
        b+=betas[i]*sigma_prev[-i]**2
    return np.sqrt(a+b)

def Hessian_1(alphas,betas,r,sigma):
    """
    Computes the Hessian matrix of the log-likelihood of a GARCH(q,p) with respect to the vector
    where vectors alphas and betas are concatenated.

    Parameters:
    alphas: A ((p+1)x1) vector.
    betas: A (qx1) vector.
    r: A (Nx1) vector.
    sigma: A (Nx1) vector.
    
    Returns: hess, a ((p+1+q)x(p+1+q)) matrix.
    """
    r_new = [r[i] for i in range(len(r)-1,-1,-1)]
    sigma_new = [sigma[i] for i in range(len(sigma)-1,-1,-1)]
    r_i = np.append(r_new,1)
    P = np.concatenate(alphas, betas) #Combine alphas and betas into one parameter vector!
    K = np.concatenate(r_i[:len(alphas)], sigma_new[:len(betas)]) #Combine r_i and sigma into one vector. 
    b=1/(np.transpose(P)@K)
    hess = -(len(r))/2 * (1/b)**(-2) * np.transpose(K)@K + 1/4 * sum(np.square(r))*b**(-3) * np.transpose(K)@K
    return hess