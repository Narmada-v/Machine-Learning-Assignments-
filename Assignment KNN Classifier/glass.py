# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:58:22 2023

@author: narma
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
glass = pd.read_csv('C:/Users/narma/Downloads/glass.csv')
glass

glass['Type'].value_counts()
glass.info()
glass.describe()
glass[glass.duplicated()].shape
glass[glass.duplicated()]
df = glass.drop_duplicates()
df
corr = df.corr()

import seaborn as sns
import matplotlib.pyplot as plt
#pairwise plot of all the features
sns.pairplot(df,hue='Type')
plt.show()

X= df.iloc[:,0:9]

array= X.values

from sklearn.preprocessing import StandardScaler
# Normalization function
SS = StandardScaler().fit(array)
SS_X =SS.transform(array)


df_knn = pd.DataFrame(SS_X,columns=df.columns[:-1])

x= df_knn
y= df['Type']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3,random_state=45)
x_train
x_test
y_train
y_test

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train,y_train)

#Predicting on test data
preds =KNN.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() 

pd.crosstab(y_test,preds) 

print("Accuracy", accuracy_score(y_test,preds)*100)
KNN.score(x_train,y_train)
print(classification_report(y_test,preds))


from sklearn.model_selection import GridSearchCV
n_neighbors = np.array(range(1,15))
grid = dict(n_neighbors=n_neighbors)
KNN = KNeighborsClassifier()
grid = GridSearchCV(estimator=KNN, param_grid=grid)
grid.fit(x, y)

print(grid.best_score_)
print(grid.best_params_)


k_values = np.arange(1,25)
train_accuracy = []
test_accuracy = []

for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))
# Plot
plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
