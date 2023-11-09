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

folder = r"C:\Users\AGSAS2291\OneDrive - Framlent AS\Dokumenter\Tidsrekker\Prosjekt\TMA4285-Project"
file = "Data1997-2022.csv"

df = pd.read_csv(os.path.join(folder,file),sep=",")

cpi_diff12 = df['CPI'].diff(12)
exog = df[['Monthly salary', 'Policy rate', 'Unemployed']]

model1 = ARIMA(df["CPI"],exog=exog,order=(1,12,3))
result = model1.fit()
result.summary()


ARX = model.ARMAX(2, 12, 0, 3,cpi_diff12[12:], exog[12:])
ARX.fit()
ARX.summary()
ARX.pythonSolution()


