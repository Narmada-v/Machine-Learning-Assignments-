# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:51:12 2023

@author: narma
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score

train = pd.read_csv('C:/Users/narma/Downloads/naives bayes assign/SalaryData_Train.csv')
train
test = pd.read_csv('C:/Users/narma/Downloads/naives bayes assign/SalaryData_Test.csv')
test
#===================================================================================
train.info()
train.describe()
test.info()
test.describe()
train[train.duplicated()].shape
train[train.duplicated()]
Train =train.drop_duplicates()
Train
Train.isnull().sum().sum()

test[test.duplicated()].shape
test[test.duplicated()]
Test=test.drop_duplicates()
Test
Test.isnull().sum().sum()
#===============================================================================
Train['Salary'].value_counts()
Test['Salary'].value_counts()
pd.crosstab(Train['occupation'],Train['Salary'])
pd.crosstab(Train['workclass'],Train['Salary'])
pd.crosstab(Train['workclass'],Train['occupation'])

#==================================================================================

sns.countplot(x='Salary',data= Train)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Train['Salary'].value_counts()


sns.countplot(x='Salary',data= Test)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Test['Salary'].value_counts()


sns.scatterplot(Train['occupation'],Train['workclass'],hue=Train['Salary'])

#================================================================================

pd.crosstab(Train['Salary'],Train['education']).mean().plot(kind='bar')

pd.crosstab(Train['Salary'],Train['occupation']).mean().plot(kind='bar')

pd.crosstab(Train['Salary'],Train['workclass']).mean().plot(kind='bar')

pd.crosstab(Train['Salary'],Train['sex']).mean().plot(kind='bar')

pd.crosstab(Train['Salary'],Train['relationship']).mean().plot(kind='bar')

string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

#Label encoding=======================================================================
LE = LabelEncoder()
for i in string_columns:
        Train[i]= LE.fit_transform(Train[i])
        Test[i]=LE.transform(Test[i])

Train

Test


##Capturing the column names=============================================================
colnames = Train.columns
colnames

len(colnames)

Train

Test

Test['maritalstatus'].value_counts()

#train test split=======================================================================
x_train = Train[colnames[0:13]].values
y_train = Train[colnames[13]].values
x_test = Test[colnames[0:13]].values
y_test = Test[colnames[13]].values


##Normalmization=====================================================================
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


x_train

x_test

y_train

y_test

x_train = norm_func(x_train)
x_test =  norm_func(x_test)


#naive bayes model on training data set================================================ 

from sklearn.naive_bayes import MultinomialNB as MB

NB=MB()
train_pred_multi=NB.fit(x_train,y_train).predict(x_train)
test_pred_multi=NB.fit(x_train,y_train).predict(x_test)

train_acc_multi=np.mean(train_pred_multi==y_train)
train_acc_multi

test_acc_multi=np.mean(test_pred_multi==y_test)
test_acc_multi

#Confusion Matrix=====================================================================
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred_multi)

#matrix
confusion_matrix

#accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_pred_multi))

#GaussianNB==========================================================================
from sklearn.naive_bayes import GaussianNB as GB
G_model=GB()
train_pred_gau=G_model.fit(x_train,y_train).predict(x_train)
test_pred_gau=G_model.fit(x_train,y_train).predict(x_test)

## train accuracy 
train_acc_gau=np.mean(train_pred_gau==y_train)
train_acc_gau 

## test acuracy
test_acc_gau=np.mean(test_pred_gau==y_test)
test_acc_gau


#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred_gau)
confusion_matrix

#accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_pred_gau))
#======================================================================================















