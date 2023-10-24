import numpy as np
import pandas as pd

# dfY=pd.read_csv("Data, response.csv",sep=";")
# dfY=dfY.rename(columns={"Unnamed: 0":"Year"})
# dfY=dfY.drop(["Y-avg2","Year"],axis=1)
# cols=dfY.columns.to_list()
# cols=cols[::-1]
# dfY=dfY[cols]
# dfY=dfY.stack(level=0)
# dfY=dfY.iloc[3:]
# dfY=dfY.iloc[::-1]

df1=pd.read_csv("ManedsLonn.csv",sep=";")
print(df1.head(n=20))