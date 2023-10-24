import numpy as np
import matplotlib.pyplot as plt


y = np.arange(21)

#ACF
def acf(y,max_lag):
    """
    Calculates the autocorrelation function for time series y , from lag 0 to lag max_lag by using
    definitions 1.14-1.1.5 (Shumway & Stoffer, 2017, p. 27-28).

    parameter y: Time Series.
    parameter max_lag: Maximum Lag.
    Returns: a vector of length max_lag, each element j is the ACF of corresponding lag j.
    """
    y_len = len(y)
    #The mean of y
    y_mean = np.mean(y)
    acfs = np.zeros(max_lag+1)
    end = max_lag+1
    #Variance
    var = sum((y-y_mean)**2)
    #Estimating ACs for lag 1...h
    for h in range(end):
        acf_h = sum((y[h:]-y_mean)*(y[:(y_len-h)]-y_mean))/var
        acfs[h] = acf_h
    
    return(acfs)

my_acf = acf(y, 21)

def acf_plot(x):
    """
    Creates an ACF-plot from a vector of ACFs, by using property 1.2 (Shumway & Stoffer, 2017, p. 28).

    param x: A vector of ACFs.
    Returns: Nothing.
    """
    xc = range(len(x))
    for j in xc:
        plt.plot((xc[j], xc[j]), (0, x[j]), "b-")
    plt.plot((0, xc[-1]), (0,0), "k-")
    plt.plot((0, xc[-1]), (2*1/np.sqrt(len(xc)), 2*1/np.sqrt(len(xc))), "c--")
    plt.plot((0, xc[-1]), (-2/np.sqrt(len(xc)), -2/np.sqrt(len(xc))), "c--")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title("Autocorrelation function (ACF)")
    plt.show()

#my_acf = acf(y, 21)
#acf_plot(my_acf)

#PACF
def pacf(y, max_lag):
    acfs = acf(y, max_lag)
    thetas = np.zeros((max_lag+1, max_lag+1))
    #Calculating PACFs.
    for i in range(1, max_lag+1):
        numerator = acfs[i]
        denominator = 1
        for k in range(1, i):
            if k == i-1:
                numerator -= thetas[k,k]*acfs[i-k]
                denominator -= thetas[k,k]*acfs[k]
            else:
                thetas[i-1, k] = thetas[i-2, k]-thetas[i-1, i-1]*thetas[i-2, i-1-k]
                numerator -= thetas[i-1, k]*acfs[i-k]
                denominator -= thetas[i-1, k]*acfs[k]
        thetas[i,i] = numerator/denominator
    pacfs = np.diagonal(thetas)
    return pacfs

def pacf_plot(x):
    xc = range(len(x))
    for j in xc:
        plt.plot((xc[j], xc[j]), (0, x[j]), "b-")
    plt.plot((0, xc[-1]), (0,0), "k-")
    plt.plot((0, xc[-1]), (2*1/np.sqrt(len(xc)), 2*1/np.sqrt(len(xc))), "c--")
    plt.plot((0, xc[-1]), (-2/np.sqrt(len(xc)), -2/np.sqrt(len(xc))), "c--")
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    plt.title("Partial Autocorrelation function (PACF)")
    plt.show()

#my_pacf = pacf(y, 21)
#pacf_plot(my_pacf)
