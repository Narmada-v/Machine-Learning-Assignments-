# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:45:58 2023

@author: narma
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Load the dataset
df = pd.read_csv('C:/Users/narma/Downloads/random forest/Company_Data.csv')
df
df.head()
df.info()
list(df)
#LABEL ENCODING===================================================================
# Convert categorical variables to numerical using dummy variables
df = pd.get_dummies(df, columns=['Urban', 'US'], drop_first=True)

# Map 'ShelveLoc' values to numerical values
df['ShelveLoc'] = df['ShelveLoc'].map({'Good': 1, 'Medium': 2, 'Bad': 3})

#features (X) and target (y)
X = df.drop('Sales', axis=1)
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

#Mean Absolute Error (MAE) and Accuracy==================================================
mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
accuracy = 100 - mape * 100

print(f"Mean Absolute Error: {round(mae, 2)}")
print(f"Accuracy: {round(accuracy, 2)}%")


#Feature Importance Analysis===========================================================
importances = rf.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Visualize feature importances
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


#Bagging====================================================================================
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
base_model = DecisionTreeRegressor(random_state=42)
bagging_model = BaggingRegressor(base_model, n_estimators=100, random_state=42)

bagging_model.fit(X_train, y_train)

bagging_predictions = bagging_model.predict(X_test)

bagging_mae = mean_absolute_error(y_test, bagging_predictions)
bagging_mape = mean_absolute_percentage_error(y_test, bagging_predictions)
bagging_accuracy = 100 - bagging_mape * 100

print(f"Bagging Mean Absolute Error: {round(bagging_mae, 2)}")
print(f"Bagging Accuracy: {round(bagging_accuracy, 2)}%")
