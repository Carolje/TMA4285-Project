import numpy as np
import pandas as pd
import plots

# dfY=pd.read_csv("Data, response.csv",sep=";")
# dfY=dfY.rename(columns={"Unnamed: 0":"Year"})
# dfY=dfY.drop(["Y-avg2","Year"],axis=1)
# cols=dfY.columns.to_list()
# cols=cols[::-1]
# dfY=dfY[cols]
# dfY=dfY.stack(level=0)
# dfY=dfY.iloc[3:]
# dfY=dfY.iloc[::-1]

# df1=pd.read_csv("ManedsLonn.csv",sep=";")
# df1=df1.T
# df11=pd.DataFrame(index=range(df1.shape[0]*12),columns=range(1))
# for i in range(df1.shape[0]):
#     df11.iloc[i*12:i*12+12,0]=df1.iloc[i,0]

# df2=pd.read_csv("Styringsrente.csv")
# #print(df2.tail(n=20))

# df3=pd.read_csv("Arbeidsledighet-prosent.csv")
# df3=df3.drop("Year",axis=1)
# cols=df3.columns.to_list()
# cols=cols[::-1]
# df3=df3[cols]
# df3=df3.stack(level=0)
# df3=df3.iloc[::-1]
# #print(dfY.shape)
# #print(df11.shape)
# #print(df2.shape)
# #print(df3.shape)
# #print(df11)
# df2=df2.iloc[130:442,]
# df3=df3.iloc[72:,]
# dfY=dfY.iloc[804:1116,]


# df=pd.DataFrame(index=range(312),columns=range(4))
# df.iloc[:,0]=dfY
# df.iloc[:,1]=df11
# #print(df2)
# df.iloc[:,2]=df2.iloc[:,1]
# df.iloc[:,3]=df3
# df=df.rename(columns={0:"CPI",1:"Monthly salary",2:"Policy rate",3:"Unemployed"})
# #print(df.head())
# df.to_csv("Data1997-2022.csv")

df=pd.read_csv("Data1997-2022.csv")
acf_CPI=plots.acf(np.array(df.iloc[:,1]),24)
#plots.acf_plot(acf_CPI)
pacf_CPI=plots.pacf(np.array(df.iloc[:,1]),24)
#print(pacf_CPI)
#plots.pacf_plot(pacf_CPI)

def differencing(covariate,n):
    cpi=np.array(covariate)
    cpi1=cpi[0:-n]
    cpi2=cpi[n:]
    cpi_diff=cpi1-cpi2
    return cpi_diff
