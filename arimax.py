# Non Kalman Filter version as i do not think i need to make a observation equation
import numpy as np


class ARMAX:
    def __init__(self, p, q, num_exo, lag_p):
      # TODO: Initialize phi parameters
        self.phi = np.zeros(p)
        self.lag_phi = lag_p
      # TODO: Initialize theta parameters
        self.theta = np.zeros(q)
      # TODO: Initalize vector for exogenous parameters
        self.beta = np.zeros(num_exo)
        pass
   
    def fit(self, data, exo_data, stop_len):
       err = np.inf
       while(err > stop_len):
           # TODO : A gradient descent step
            np.norm
           # Calculate the residuals

           # Update the variables along the gradient

           # calculate length of total step

    def predict(self, data, exo_data):
        """
        Makes a prediction at times t
        """

        x_t = np.dot(self.beta, exo_data) + np.dot(self.phi, data)
        