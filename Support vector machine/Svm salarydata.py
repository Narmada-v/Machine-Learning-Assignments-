# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:52:40 2023

@author: narma
"""

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#=====================================================================================
Train = pd.read_csv('C:/Users/narma/Downloads/SVM/SalaryData_Train(1).csv')
Test = pd.read_csv('C:/Users/narma/Downloads/SVM/SalaryData_Test(1).csv')
Train
Test

Train.info()
Train.describe()
Test.info()
Test.describe()
Train[Train.duplicated()].shape
Train[Train.duplicated()]
Train =Train.drop_duplicates()
Train
Train.isnull().sum().sum()

Test[Test.duplicated()].shape
Test[Test.duplicated()]
Test=Test.drop_duplicates()
Test
Test.isnull().sum().sum()
#====================================================================================
Train['Salary'].value_counts()
Test['Salary'].value_counts()
pd.crosstab(Train['occupation'],Train['Salary'])
pd.crosstab(Train['workclass'],Train['Salary'])
pd.crosstab(Train['workclass'],Train['occupation'])


# Visualize data
sns.countplot(x='Salary', data=Train)
plt.xlabel('Salary')
plt.ylabel('Count')
plt.show()

sns.countplot(x='Salary',data= Test)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Test['Salary'].value_counts()


sns.scatterplot(Train['occupation'],Train['workclass'],hue=Train['Salary'])

pd.crosstab(Train['Salary'],Train['education']).mean().plot(kind='bar')

pd.crosstab(Train['Salary'],Train['occupation']).mean().plot(kind='bar')

pd.crosstab(Train['Salary'],Train['workclass']).mean().plot(kind='bar')

pd.crosstab(Train['Salary'],Train['sex']).mean().plot(kind='bar')

pd.crosstab(Train['Salary'],Train['relationship']).mean().plot(kind='bar')

# scatter matrix to observe relationship between every colomn attribute. 
pd.plotting.scatter_matrix(Train,
                                       figsize= [20,20],
                                       diagonal='hist',
                                       alpha=1,
                                       s = 300,
                                       marker = '.',
                                       edgecolor= "black")
plt.show()

#===================================================================================

string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
##Preprocessing  categorical variables
number = LabelEncoder()
for i in string_columns:
        Train[i]= number.fit_transform(Train[i])
        Test[i]=number.fit_transform(Test[i])
Train

Test

#column names
colnames = Train.columns
colnames

Train

Test
#Train test split====================================================================
x_train = Train[colnames[0:13]]
y_train = Train[colnames[13]]
x_test = Test[colnames[0:13]]
y_test = Test[colnames[13]]
##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
x_train = norm_func(x_train)
x_test =  norm_func(x_test)


#SVM model=============================================================================
model_linear = SVC(kernel = "linear",random_state=40,gamma=0.1,C=1.0)
model_linear.fit(x_train,y_train)

SVC(gamma=0.1, kernel='linear', random_state=40)

pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test) 


# Kernel = poly===========================================================================
model_poly = SVC(kernel = "poly",random_state=40,gamma=0.1,C=1.0)
model_poly.fit(x_train,y_train)

pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test)


# kernel = rbf============================================================================
model_rbf = SVC(kernel = "rbf",random_state=40,gamma=0.1,C=1.0)
model_rbf.fit(x_train,y_train)

pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test)


#'sigmoid'=============================================================================
model_sig = SVC(kernel = "sigmoid",random_state=40,gamma=0.1,C=1.0)
model_sig.fit(x_train,y_train)

pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test)


#SVM WITH gridsearch

# kernel = rbf=========================================================================
clf= SVC()
parma_grid = [{'kernel' : ["rbf"],'random_state':[40],'gamma':[0.1],'C':[1.0]}]

gsv = GridSearchCV(clf,parma_grid,cv=10)

gsv.fit(x_train,y_train)

gsv.best_params_ , gsv.best_score_

clf = SVC(C= 15, gamma = 50)
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)

confusion_matrix(y_test, y_pred)

# kernel = linear=======================================================================
clf= SVC()
parma_grid = [{'kernel' : ["linear"],'random_state':[40],'gamma':[0.1],'C':[1.0]}]
gsv = GridSearchCV(clf,parma_grid,cv=10)

gsv.fit(x_train,y_train)

gsv.best_params_ , gsv.best_score_

clf = SVC(C= 15, gamma = 50)
clf.fit(x_train , y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)

confusion_matrix(y_test, y_pred)


# kernel = poly=======================================================================
clf= SVC()
parma_grid = [{'kernel' : ["poly"],'random_state':[40],'gamma':[0.1],'C':[1.0]}]

gsv = GridSearchCV(clf,parma_grid,cv=10)

gsv.fit(x_train,y_train)

gsv.best_params_ , gsv.best_score_

clf = SVC(C= 15, gamma = 50)
clf.fit(x_train , y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)    
#===================================================================================           