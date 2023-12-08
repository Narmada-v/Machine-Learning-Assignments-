# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:08:44 2023

@author: narma
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:08:57 2019

@author: Hello
"""

import pandas as pd 
import numpy as np 
import seaborn as sns

df = pd.read_csv("C:/Users/narma/Downloads/SVM/forestfires.csv")
df
df.describe()
df.info()
##Dropping the month and day columns
df.drop(["month","day"],axis=1,inplace =True)
df
##Normalising the data as there is scale difference
x =df.iloc[:,0:28]
y= df.iloc[:,28]

def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

x_norm = norm_func(x)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, stratify = y)

model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test)*100

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test)*100

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test)*100 

#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test)*100 
