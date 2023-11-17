# TMA4285-Project
This a code to a project in the the subject TMA4285 Time Series at NTNU. The task was to implment versions of ARMAX and GARCH and fit these to data about the Consumer price index in Norway. 

In the file dataTransformations.py a lot of the commented out code describes how we made the dataFrame file with our response and covariates. The resulting data frame is stored in Data1997-2022.csv. dataTransformations.py also includes a function for performing differncing the data. 

In the file plots.py we have implemented our own functions to calculate ACF and PACF along with functions to plot the ACF and PACF with confidence intervals.

In the file garchFuncs.py we have functions used to fit a Garch(p,q) model. The file garch.py contains our function to fit the GARCH model along with the models we have fitted. Running the garch.py file will reproduce our results. 

In the file arimax.py is a class to fit ARMAX models and get a summary of the fitted model. The file test_arima.py is used to fit and test the armax model. 
