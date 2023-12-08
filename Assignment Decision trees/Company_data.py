# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:28:40 2023

@author: narma
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:/Users/narma/Downloads/decision trees/Company_Data.csv')
df
df.head()
df.info()


#LABEL ENCODING===================================================================
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Urban_Yes'] = LE.fit_transform(df['Urban_Yes'])
df['US_Yes'] = LE.fit_transform(df['US_Yes'])
df['Urban_Yes'] 
df['US_Yes']

# Convert 'ShelveLoc' categorical variable into numerical using mapping
df['ShelveLoc'] = df['ShelveLoc'].map({'Good': 1, 'Medium': 2, 'Bad': 3})
df['ShelveLoc']

# Separate features and target variable
x = df.iloc[:, 0:6]  # Features
y = df['ShelveLoc']  # Target variable
x
y
# Splitting data into training and testing sets===================================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Decision Tree Classifier using Entropy Criteria===============================
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy', max_depth=3)
DT.fit(x_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Assuming x contains your DataFrame features
feature_names_list = list(x.columns)

plt.figure(figsize=(10, 6))
plot_tree(DT, feature_names=feature_names_list, class_names=['1', '2', '3'], filled=True)
plt.show()
Y_preds = DT.predict(x_test)
Y_preds
#accuracy
accuracy = np.mean(Y_preds == y_test)
accuracy
# Confusion Matrix
conf_matrix = pd.crosstab(y_test, Y_preds)
conf_matrix
# Printing accuracy and confusion matrix
print(f"Accuracy: {accuracy:.3f}")
print(conf_matrix)
#=========================================================================================
# Building Decision Tree Classifier using Gini Criteria
from sklearn.tree import DecisionTreeClassifier
DT_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
DT_gini.fit(x_train, y_train)

Y_pred = DT.predict(x_test)
accuracy_gini = np.mean(Y_pred == y_test)
print(f"Accuracy using Gini: {accuracy_gini:.3f}")
#=========================================================================================
# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
X = df.iloc[:, 0:3]
X
y = df.iloc[:, 3]
y
# Splitting data into training and testing sets for regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Decision Tree Regression Model
reg_model = DecisionTreeRegressor()
reg_model.fit(X_train, y_train)

# Accuracy for regression
reg_accuracy = reg_model.score(X_test, y_test)
print(f"Regression Accuracy: {reg_accuracy:.3f}")
