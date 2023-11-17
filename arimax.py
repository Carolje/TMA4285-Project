# Non Kalman Filter version as i do not think i need to make a observation equation
import numpy as np
import numpy.linalg as la
from statsmodels.tsa.arima.model import ARIMA
import scipy.optimize as opt
import statistics as s
import scipy.stats as st


class ARMAX:
    def __init__(self, p, q, y, exo_data):
      # overall
      self.p = p
      self.q = q
      self.y = y
      self.exog = exo_data
      self.init_params = np.array([])
      self.res = None
      
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
        sigma2 = var*var            
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
        for t in range(1,len(x_tt)):
            Sig_t = obsv_matrix @ P_tt1[t].reshape((2,2)) @ obsv_matrix.T
            res_t = y[t] - obsv_matrix @ x_tt1[t] - exog_matrix @ exog[t] 
            neg_ll += np.log(np.abs(Sig_t)) + res_t**2/ np.abs(Sig_t)

        print(neg_ll)
        return neg_ll

    def fit_kalman(self, evo, var, beta):
         self.init_params = np.concatenate((np.append(evo, var), beta), axis = 0)       
         self.res = opt.minimize(self.kalman_log_likelihood, self.init_params, method ='BFGS')
         return self.res
    
    def calc_aic(self):
        return self.res.fun + 2*len(self.res.x)

    
    def summary(self):
        """
        Calulate the rest of the results
        """
        std_errors = self.res.hess_inv.diagonal()
        z_val = self.res.x/std_errors
        p_val = (1-st.norm.cdf(abs(z_val)))*2
        lower = self.res.x - st.norm.ppf(0.95)*std_errors 
        upper = self.res.x + st.norm.ppf(0.95)*std_errors 
        aic = self.calc_aic()
        
        
        """
        Prints the results 
        """
        print('{:^51}'.format("Results"))
        print("="*51)
        print('{:<25}'.format("Dep. Variable:"),'{:>25}'.format(self.y.name))
        print('{:<25}'.format("Model:"),'{:>17}'.format("ARIMAX("),self.p,",",self.q,")")
        print('{:<25}'.format("No. Observations:"),'{:>25}'.format(len(self.y)))
        print('{:<25}'.format("AIC"),'{:>25.3e}'.format(aic))
        print('{:<25}'.format("BIC"),'{:>25}'.format("BICvalue"))
        print('{:<25}'.format("Log Likelihood"),'{:>25}'.format("Logvalue"))
        print("="*91)
        print('{:>25}'.format("init.values"),'{:>10}'.format("coef"),'{:>10}'.format("std.err"),'{:>10}'.format("z"),
              '{:>10}'.format("P>|z|"),'{:>10}'.format("[0.025"),'{:>10}'.format("0.975]"))
        print("-"*91)
        # exogenious terms
        for (i,x) in enumerate(self.exog.columns):
            init = self.init_params[3:3+len(self.exog.columns)]
            res = self.res.x[3:3+len(self.exog.columns)]
            std = std_errors[3:3+len(self.exog.columns)]
            z = z_val[3:3+len(self.exog.columns)]
            p = p_val[3:3+len(self.exog.columns)]
            u = upper[3:3+len(self.exog.columns)]
            l = lower[3:3+len(self.exog.columns)]
            print('{:<14}'.format(x),'{:>10.3e}'.format(init[i]),
                  '{:>10.3e}'.format(res[i]),'{:>10.3e}'.format(std[i]),
                  '{:>10.3e}'.format(z[i]),'{:>10.3e}'.format(p[i]),
                  '{:>10.3e}'.format(l[i]),'{:>10.3e}'.format(u[i]))
        # ar terms
        for i in range(self.p):
            init = self.init_params[:2]
            res = self.res.x[:2]
            std = std_errors[:2]
            z = z_val[:2]
            p = p_val[:2]
            u = upper[:2]
            l = upper[:2]
            print('{:<14}'.format("ar.L"+str(i+1)),'{:>10.3e}'.format(init[i]),
                  '{:>10.3e}'.format(res[i]),'{:>10.3e}'.format(std[i]),
                  '{:>10.3e}'.format(z[i]),'{:>10.3e}'.format(p[i]),
                  '{:>10.3e}'.format(l[i]),'{:>10.3e}'.format(u[i]))
        # ma terms
        for i in range(self.q):
            init = self.init_params[3+len(self.exog.columns):]
            res = self.res.x[3+len(self.exog.columns):]
            std = std_errors[3+len(self.exog.columns):]
            z = z_val[3+len(self.exog.columns):]
            p = p_val[3+len(self.exog.columns):]
            u = upper[3+len(self.exog.columns):]
            l = lower[3+len(self.exog.columns):]
            print('{:<14}'.format("ma.L"+str(i+1)),'{:>10.3e}'.format(init[i]),
                  '{:>10.3e}'.format(res[i]),'{:>10.3e}'.format(std[i]),
                  '{:>10.3e}'.format(z[i]),'{:>10.3e}'.format(p[i]),
                  '{:>10.3e}'.format(l[i]),'{:>10.3e}'.format(u[i]))
        #sigma
        print('{:<14}'.format("sigma"),'{:>10.3e}'.format(self.init_params[2]),
              '{:>10.3e}'.format(self.res.x[2]),'{:>10.3e}'.format(std_errors[2]),
              '{:>10.3e}'.format(z_val[2]),'{:>10.3e}'.format(p_val[2]),
              '{:>10.3e}'.format(lower[2]),'{:>10.3e}'.format(upper[2]))
        print("="*91)
        
        
        
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
        



        