# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:12:06 2023

@author: AGSAS2291
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import arimax as model
import statistics as s

folder = r"C:\Users\AGSAS2291\OneDrive - Framlent AS\Dokumenter\Tidsrekker\Prosjekt\TMA4285-Project"
file = "Data1997-2022.csv"

df = pd.read_csv(os.path.join(folder,file),sep=",")
df['Intercept'] = np.ones(len(df))

cpi_diff12 = df['CPI'].diff(12)
exog = df[['Intercept','Monthly salary', 'Policy rate', 'Unemployed']]

model1 = ARIMA(df["CPI"],exog=exog,order=(1,12,3))
result = model1.fit()
result.summary()


ARX = model.ARMAX(2, 0,cpi_diff12[12:], exog[12:])
ARX.pythonSolution()
evo = np.array([-1.6242,-0.8331])
var = 72
beta = np.array([1.101e-8,-1e-4,0.0251,0.0449])
opt = ARX.fit_kalman(evo, var, beta)
ARX.summary()
print(opt.hess_inv)