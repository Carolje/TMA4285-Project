# Non Kalman Filter version as i do not think i need to make a observation equation
import numpy as np
import numpy.linalg as la
from statsmodels.tsa.arima.model import ARIMA
import scipy.optimize as opt
import statistics as s


class ARMAX:
    def __init__(self, p, q, num_exo, y, exo_data,alpha_phi=0.0004, alpha_beta=0.4*10**(-12), stop_len=10**(-6)):
      # overall
      self.p = p
      self.q = q
      self.y = y
      self.exog = exo_data
      # fit()
      # TODO: Initialize phi parameters
      self.phi = np.zeros(p)
      self.alpha_phi = alpha_phi
      self.alpha_beta = alpha_beta
      self.stop_len = stop_len
      # TODO: Initialize theta parameters
      self.theta = np.zeros(q)
      # TODO: Initalize vector for exogenous parameters
      self.beta = np.zeros(num_exo)
      pass

    def kalman_log_likelihood(self, params):
        y = np.array(self.y)
        exog = np.array(self.exog)
        # params = [phi1, phi2, sigma, beta0, beta1, beta2, beta3]
        evo, var, beta = params[0:2], params[2], params[3:]
        # Initial conditions
        x_0, P_0 = np.ones(self.p)*s.mean(y), np.zeros((self.p,self.p))
        np.fill_diagonal(P_0, s.variance(y))
        # F 2x2
        evo_matrix = np.zeros((self.p,self.p))        
        evo_matrix[0,:] = evo
        evo_matrix[1:,:-1] = np.diag(np.ones(self.p-1))
        # A 1x2
        obsv_matrix = np.zeros(self.p)      
        obsv_matrix[0] = 1
        # v_t - white noise 1x1
        sigma2 = var            
        # H 1x4
        exog_matrix = beta
        # G 2x1
        noise_matrix = np.array([np.sqrt(sigma2),0])    
        # Q 2x2
        noise_state = np.zeros((2,2))
        noise_state[0,0] = sigma2

        x_tt = np.zeros((len(y),2))
        x_tt1 = np.zeros((len(y),2))
        P_tt = np.zeros((len(y),4))
        P_tt1 = np.zeros((len(y),4))
        x_tt[0] = x_0
        P_tt[0] = P_0.reshape((1,4))
        
        for t in range(1,len(x_tt)):
            # Kalman Filter
            #prediction
            x_tt1[t] = evo_matrix @ x_tt[t-1]
            P_tt1[t] = (evo_matrix @ P_tt[t-1].reshape((2,2)) @ evo_matrix.T + noise_state).reshape((1,4))
            #filter
            K_t = P_tt1[t].reshape((2,2)) @ obsv_matrix.T * 1/(obsv_matrix @ P_tt1[t].reshape((2,2)) @ obsv_matrix.T)
            x_tt[t] = x_tt1[t] + K_t * (y[t] - obsv_matrix @ x_tt1[t] - exog_matrix @ exog[t])
            P_tt[t] = ((np.identity(2) - K_t @ obsv_matrix) @ P_tt1[t].reshape((2,2))).reshape((1,4))
            
        # Likelihood and minimize
        neg_ll = 0
        for t in range(len(x_tt)):
            Sig_t = obsv_matrix @ P_tt1[t].reshape((2,2)) @ obsv_matrix.T
            res_t = y[t] - obsv_matrix @ x_tt1[t] - exog_matrix @ exog[t] 
            # if Sig_t == np.nan:
            #     print('skraaa')
            neg_ll += np.log(Sig_t) + res_t**2/ Sig_t

        print(neg_ll)
        return neg_ll

    def fit_kalman(self, evo, var, beta):
         init_params = np.concatenate((np.append(evo, var), beta), axis = 0)       
         res = opt.minimize(self.kalman_log_likelihood, init_params, method ='BFGS')
         return res
   
    def fit(self):
        step_len = np.inf
        while(step_len > self.stop_len):
           # TODO : Implement for MA
           
           
           # AR
           if self.p != 0:
               # Calculate the residuals
               res = self.predict() - self.y[self.p-1:]
                
               # Update the variables along the gradient
               phi_step = np.dot(res, self.y[self.p-1:])
               beta_step = np.dot(res, self.exog[self.p-1:])
               # calculate length of total step
               step_len = self.alpha_phi*la.norm(phi_step) + self.alpha_beta*la.norm(beta_step)
    
               self.phi -= self.alpha_phi*phi_step
               self.beta -= self.alpha_beta*beta_step
           #print(self.beta)
           #print(self.phi)
        print("final")
        print(self.beta)
        print(self.phi)

    def predict(self):
        """
        Makes a prediction at times t
        """
        ar_term = np.convolve(self.y, self.phi, 'valid')
        exog_term = np.dot(self.exog[self.p-1:], self.beta)
        x_t = ar_term + exog_term 
        # These are predictions not containg the first p values

        
        return x_t
    
    def summary(self):
        """
        Prints the results 
        """
        print('{:^80}'.format("Results"))
        print("="*80)
        print('{:<25}'.format("Dep. Variable:"),'{:>25}'.format(self.y.name))
        print('{:<25}'.format("Model:"),'{:>17}'.format("ARIMAX("),self.p,",",self.q,")")
        print('{:<25}'.format("No. Observations:"),'{:>25}'.format(len(self.y)))
        print('{:<25}'.format("AIC"),'{:>25}'.format("AICvalue"))
        print('{:<25}'.format("BIC"),'{:>25}'.format("BICvalue"))
        print('{:<25}'.format("Log Likelihood"),'{:>25}'.format("Logvalue"))
        print("="*80)
        print('{:>25}'.format("coef"),'{:>10}'.format("std.err"),'{:>10}'.format("z"),
              '{:>10}'.format("P>|z|"),'{:>10}'.format("[0.025"),'{:>10}'.format("0.975]"))
        print("-"*80)
        # exogenious terms
        for (i,x) in enumerate(self.exog.columns):
            print('{:<14}'.format(x),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589))
        # ar terms
        for i in range(self.p):
            print('{:<14}'.format("ar.L"+str(i+1)),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589))
        # ma terms
        for i in range(self.q):
            print('{:<14}'.format("ma.L"+str(i+1)),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589))
        print("="*80)
        
        
        
    def pythonSolution(self):
        """
        Prints result

        Returns
        -------
        The solution ARIMA() from python library gives

        """
        model1 = ARIMA(self.y,exog=self.exog,order=(self.p,12,self.q))
        result = model1.fit()
        print(result.summary())
        



        