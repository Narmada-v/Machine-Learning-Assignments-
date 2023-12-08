# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:02:09 2023

@author: narma
"""

#IMPORTING THE CSV FILE=========================================================
import pandas as pd
df = pd.read_csv('C:/Users/narma/Downloads/Zoo.csv')
df.info()
df.describe()
df['animal name'].value_counts()
list(df)

#check if there are duplicates in animal_name===================================
duplicates = df['animal name'].value_counts()
duplicates[duplicates > 1]

frog = df[df['animal name'] == 'frog']
frog

# observation: find that one frog is venomous and another one is not 
# change the venomous one into frog2 to seperate 2 kinds of frog=============== 
df['animal name'][(df['venomous'] == 1 )& (df['animal name'] == 'frog')] = "frog2"
df['venomous'].value_counts()
df.head(27)

# finding Unique value of hair and plotting=====================================================
color_list = [("red" if i == 1 else "blue" if i == 0 else "yellow" ) for i in df.hair]
unique_color = list(set(color_list))
unique_color

import seaborn as sns 
import matplotlib.pyplot as plt
sns.countplot(x="hair", data=df)
plt.xlabel("Hair")
plt.ylabel("Count")
plt.show()
df.loc[:,'hair'].value_counts()


# Lets see how many animals provides us milk and plotting=================================
df['milk'].value_counts()


# Lets see species wise domestic and non-domestic animals
pd.crosstab(df['type'], df['milk']).plot(kind="bar", figsize=(10, 8), title="milk providing animals");
plt.plot();

# lets find out all the aquatic animals and plotting======================================
pd.crosstab(df['type'], df['aquatic']).plot(kind="bar", figsize=(10, 8));


#DATA PARTITION==========================================================================

from sklearn.model_selection import train_test_split
X = df.iloc[:,1:16]
Y = df.iloc[:,16]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)
X_train
X_test
Y_train
Y_test 


# LABEL ENCODING=======================================================================
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Y = LE.fit_transform(Y)
Y


#KFOLD==================================================================================
from sklearn.model_selection import KFold
num_folds = 10
kfold = KFold(n_splits=10)

#KNN MODEL================================================================================
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)

#Predicting on test data
Y_preds = model.predict(X_test) 
Y_preds
pd.Series(Y_preds).value_counts()

pd.crosstab(Y_test,Y_preds) 


# Accuracy ==============================================================================
import numpy as np
np.mean(Y_preds==Y_test)

model.score(X_train,Y_train)

#ACCURACY SCORE==========================================================================
from sklearn.metrics import accuracy_score
print("Accuracy", accuracy_score(Y_test,Y_preds)*100)

#CROSS VALIDATION FOR SMALL DATA=========================================================
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean()*100)

print(results.std()*100)


#GRID SEARCH====================================================================

from sklearn.model_selection import GridSearchCV
n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)
model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)

print(grid.best_score_)
print(grid.best_params_)


k_values = np.arange(1,25)
train_accuracy = []
test_accuracy = []

for i, k in enumerate(k_values):
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train,Y_train)
    train_accuracy.append(KNN.score(X_train,Y_train))
    test_accuracy.append(KNN.score(X_test,Y_test))

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
#=====================================================================================================



