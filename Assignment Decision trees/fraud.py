# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:57:20 2023

@author: narma
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('C:/Users/narma/Downloads/decision trees/Fraud_check.csv')
df
df.head()
df.info()

# Assigning values to 'Risky' and 'Good' based on Taxable Income==========================
df['Taxable.Income'] = pd.cut(df['Taxable.Income'], bins=[0, 30000, float('inf')], labels=['Risky', 'Good'])

# Converting categorical variables to numerical using one-hot encoding
df= pd.get_dummies(df, columns=['Undergrad', 'Marital.Status', 'Urban'], drop_first=True)

# Splitting 
X = df.drop('Taxable.Income', axis=1)
X
y = df['Taxable.Income']
y

# List of columns to convert from boolean to numeric
columns_to_convert = ['Undergrad_YES', 'Marital.Status_Single', 'Urban_YES']
columns_to_convert
# Convert True to 1 and False to 0 in selected columns
X[columns_to_convert] = X[columns_to_convert].astype(int)
[columns_to_convert]
# Display the updated DataFrame
X


# Splitting the data=========================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# decision tree classifier
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy', max_depth=3)
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
y_pred
#PLOTTING
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
# Assuming x contains your DataFrame features
feature_names_list = list(X.columns)
plt.figure(figsize=(10, 6))
plot_tree(DT, feature_names=feature_names_list, class_names=['1', '2', '3'], filled=True)
plt.show()
#accuracy
accuracy = np.mean(y_pred == y_test)
accuracy
# Confusion Matrix
conf_matrix = pd.crosstab(y_test, y_pred)
conf_matrix
#=================================================================================================

# Evaluating the model
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Decision Tree Classifier using Gini Criteria
from sklearn.tree import DecisionTreeClassifier
DT_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
DT_gini.fit(X_train, y_train)

y_pred = DT.predict(X_test)
accuracy_gini = np.mean(y_pred == y_test)
print(f"Accuracy using Gini: {accuracy_gini:.3f}")

#==========================================================================================

