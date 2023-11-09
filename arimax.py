# Non Kalman Filter version as i do not think i need to make a observation equation
import numpy as np
import numpy.linalg as la
from statsmodels.tsa.arima.model import ARIMA


class ARMAX:
    def __init__(self, p, d, q, num_exo, data, exo_data,alpha_phi=0.0004, alpha_beta=0.4*10**(-12), stop_len=10**(-6)):
      # TODO: Initialize phi parameters
      self.phi = np.zeros(p)
      self.p = p
      self.d = d
      self.q = q
      self.data = data
      self.exog = exo_data
      self.alpha_phi = alpha_phi
      self.alpha_beta = alpha_beta
      self.stop_len = stop_len
      # TODO: Initialize theta parameters
      self.theta = np.zeros(q)
      # TODO: Initalize vector for exogenous parameters
      self.beta = np.zeros(num_exo)
      pass
  
   
    def fit(self):
        step_len = np.inf
        while(step_len > self.stop_len):
           # TODO : Implement for MA
           
           
           # AR
           if self.p != 0:
               # Calculate the residuals
               res = self.predict() - self.data[self.p-1:]
                
               # Update the variables along the gradient
               phi_step = np.dot(res, self.data[self.p-1:])
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
        ar_term = np.convolve(self.data, self.phi, 'valid')
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
        print('{:<25}'.format("Dep. Variable:"),'{:>25}'.format(self.data.name))
        print('{:<25}'.format("Model:"),'{:>13}'.format("ARIMAX("),self.p,",",self.d,",",self.q,")")
        print('{:<25}'.format("No. Observations:"),'{:>25}'.format(len(self.data)))
        print('{:<25}'.format("AIC"),'{:>25}'.format("AICvalue"))
        print('{:<25}'.format("BIC"),'{:>25}'.format("BICvalue"))
        print('{:<25}'.format("Log Likelihood"),'{:>25}'.format("Logvalue"))
        print("="*80)
        print('{:>25}'.format("coef"),'{:>10}'.format("std.err"),'{:>10}'.format("z"),
              '{:>10}'.format("P>|z|"),'{:>10}'.format("[0.025"),'{:>10}'.format("0.975]"))
        print("-"*80)
        # differencing term
        if self.d == 0:
            print('{:<14}'.format("const"),'{:>10.4f}'.format(4.23589),
                  '{:>10.4f}'.format(4.23589),'{:>10.4f}'.format(4.23589),
                  '{:>10.4f}'.format(4.23589),'{:>10.4f}'.format(4.23589),
                  '{:>10.4f}'.format(4.23589))
        # exogenious terms
        for (i,x) in enumerate(self.exog.columns):
            print('{:<14}'.format(x),'{:>10.3e}'.format(self.beta[i]),
                  '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589),'{:>10.3e}'.format(4.23589),
                  '{:>10.3e}'.format(4.23589))
        # ar terms
        for i in range(self.p):
            print('{:<14}'.format("ar.L"+str(i+1)),'{:>10.3e}'.format(self.phi[i]),
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
        model1 = ARIMA(self.data,exog=self.exog,order=(self.p,self.d,self.q))
        result = model1.fit()
        print(result.summary())
        



        