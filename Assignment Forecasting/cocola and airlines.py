# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:50:55 2023

@author: narma
"""

import pandas as pd
import numpy as np


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Forecasting\\CocaCola_Sales_Rawdata.csv")


df


df.describe()



import matplotlib.pyplot as plt
df.plot()



import seaborn as  sns
sns.boxplot(data =df)


df.hist()


df.plot(kind="kde")

np.array(df["Sales"])

import seaborn as sns
sns.set_theme()
rk= sns.distplot(df['Sales'],kde=True)


from pandas.plotting import lag_plot
lag_plot(df['Sales'])

# DATA PREPROCESSING
df.head()

len(df)

df['quarter'] = 0
for i in range(42):
    p=df['Quarter'][i]
    df['quarter'][i]=p[0:2]

df


df['quarter'].value_counts()

df_dummies=pd.DataFrame(pd.get_dummies(df['quarter']),columns=['Q1','Q2','Q3','Q4'])
cc=pd.concat([df,df_dummies],axis= 1)

cc


cc['t'] = np.arange(1,43)
cc['t_squared'] = cc['t']**2
cc["Sales_log"] =np.log(df['Sales'])


cc.head()


train =cc.head(32)
test =cc.tail(10)



df['Sales'].plot()


# MODELS
from sklearn.metrics import mean_squared_error


# Linear Model
import statsmodels.formula.api as smf
linear_model =smf.ols("Sales~t",data =train).fit()
linear_pred = pd.Series(linear_model.predict(test['t']))
linear_rmse =np.sqrt(mean_squared_error(np.array(test['Sales']),np.array(linear_pred)))
linear_rmse

#Quadratic Model
quad_model =smf.ols("Sales~t+t_squared",data=train).fit()
quad_pred = pd.Series(quad_model.predict(test[['t','t_squared']]))
quad_rmse =np.sqrt(mean_squared_error(np.array(test['Sales']),np.array(quad_pred)))
quad_rmse

# Exponential model
exp_model  =smf.ols("Sales_log~t",data=train).fit()
exp_pred =pd.Series(exp_model.predict(test['t']))
exp_rmse =np.sqrt(mean_squared_error(np.array(test['Sales']),np.array(exp_pred)))
exp_rmse


data = {"MODEL":pd.Series(["rmse_linear","rmse_exp","rmse_quad"]),"RMSE_Values":pd.Series([linear_rmse,exp_rmse,quad_rmse,])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# # QUESTION - 2(airlines csv file)



import pandas as pd
import numpy as np



df=pd.read_excel("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Forecasting\\Airlines+Data.xlsx")

df



df.shape


df.head()


df.info()


df.describe()

list(df)



df["months"]=df.Month.dt.strftime("%b")
df

df["year"]=df.Month.dt.strftime("%Y")
df


df.drop(["Month"],axis=1,inplace=True)



df


df.set_index(["year"])


# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
heatmap_y_month = pd.pivot_table(data=df, values="Passengers", index="year", columns="months", fill_value=0)
sns.heatmap(heatmap_y_month, annot=True, fmt="g")


# Box plot
plt.figure(figsize=(12,8))
sns.boxplot(x="months",y="Passengers",data=df)


plt.figure(figsize=(12,8))
sns.boxplot(x="year",y="Passengers",data=df)


df["t"]=np.arange(1,len(df)+1)
df["t_square"]=df["t"]**2
dummy=pd.DataFrame(pd.get_dummies(df["months"]))
df1=pd.concat([df, dummy.astype(int)],axis=1)
df1.head()


# Data partition
train=df1.head(96)
test=df1.tail(19)
df1.columns


# Forecasting Models
# linear
#linear model
import statsmodels.formula.api as smf
linear_model=smf.ols('Passengers~ t',data=df).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear=np.sqrt(np.mean(np.array(test['Passengers'])-np.array(pred_linear))**2)
rmse_linear

# Exponential
Exp = smf.ols('np.log(Passengers) ~ t', data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Passengers']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# Quadratic
Quad = smf.ols('Passengers ~ t + t_square', data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[['t', 't_square']]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Passengers']) - np.array(pred_Quad))**2))
rmse_Quad


# Additive seasonality
add_sea = smf.ols('Passengers ~ Apr + Aug + Dec + Feb + Jan + Jul + Jun + Mar + May + Nov + Oct + Sep', data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Apr', 'Aug', 'Dec',
       'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Passengers']) - np.array(pred_add_sea))**2))
rmse_add_sea


# Additive Seasonality Quadratic
add_sea_Quad = smf.ols('Passengers ~ t + t_square + Apr + Aug + Dec + Feb + Jan + Jul + Jun + Mar + May + Nov + Oct + Sep', data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['t', 't_square', 'Apr', 'Aug', 'Dec',
       'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Passengers']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# Multiplicative Seasonality
Mul_sea = smf.ols('np.log(Passengers) ~  Apr + Aug + Dec + Feb + Jan + Jul + Jun + Mar + May + Nov + Oct + Sep', data=train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test[['Apr', 'Aug', 'Dec',
       'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep']]))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Passengers']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

# Multiplicative Additive Seasonality
Mul_Add_sea = smf.ols('np.log(Passengers) ~  t + Apr + Aug + Dec + Feb + Jan + Jul + Jun + Mar + May + Nov + Oct + Sep', data=train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test[['t', 'Apr', 'Aug', 'Dec',
       'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep']]))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Passengers']) - np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea

# Compare the results
data = {"MODEL": pd.Series(["rmse_linear", "rmse_Exp", "rmse_Quad", "rmse_add_sea", "rmse_add_sea_quad", "rmse_Mult_sea", "rmse_Mult_add_sea"]),
        "RMSE_Values": pd.Series([rmse_linear, rmse_Exp, rmse_Quad, rmse_add_sea, rmse_add_sea_quad, rmse_Mult_sea, rmse_Mult_add_sea])}


data


table_rmse = pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


