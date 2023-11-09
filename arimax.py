# Non Kalman Filter version as i do not think i need to make a observation equation
import numpy as np
import numpy.linalg as la


class ARMAX:
    def __init__(self, p, q, num_exo):
      # TODO: Initialize phi parameters
        self.phi = np.zeros(p)
        self.p = p
      # TODO: Initialize theta parameters
        self.theta = np.zeros(q)
      # TODO: Initalize vector for exogenous parameters
        self.beta = np.zeros(num_exo)
        pass
   
    def fit(self, data, exo_data, alpha_phi, alpha_beta, stop_len):
       step_len = np.inf
       while(step_len > stop_len):
           # TODO : Implement for MA
            
           # Calculate the residuals
            res = self.predict(data, exo_data) - data[self.p:]
            
           # Update the variables along the gradient
            phi_step = np.dot(res, data[:-self.p])
            beta_step = np.dot(res, exo_data[self.p:])
           # calculate length of total step
            step_len = alpha_phi*la.norm(phi_step) + alpha_beta*la.norm(beta_step)

            self.phi -= alpha_phi*phi_step
            self.beta -= alpha_beta*beta_step

            # print(self.beta)
            # print(self.phi)
      #  print('final')
      #  print(self.beta)
      #  print(self.phi)
      

    def predict(self, data, exo_data):
        """
        Makes a prediction at times t
        """
        ar_term = np.convolve(data[:-1], self.phi, 'valid')   # TODO : Think about if this is right with the self.p stop on data
        exog_term = np.dot(exo_data[self.p:], self.beta)
        x_t = ar_term + exog_term 
        # These are predictions not containg the first p values
        
        return x_t



        