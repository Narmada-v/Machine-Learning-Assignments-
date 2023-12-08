# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:12:07 2023

@author: narma
"""

# Question - 1

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("C:/Users/narma/Downloads/50_Startups.csv")

# EDA==========================================================================
sns.pairplot(df)
plt.show()

# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Extracting features and target variable=======================================
X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
Y = df['Profit']

# Data Transformation=========================================================
# Label Encoding
LE = LabelEncoder()
X['State'] = LE.fit_transform(df['State'])

# Standardization
SS = StandardScaler()
SS_X= SS.fit_transform(X)

# VIF Check for MULTICOLLINEARITY=============================================
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(SS_X, i) for i in range(SS_X.shape[1])]
print("VIF:")
print(vif_data)

# Model Fitting with Model Validation Techniques==============================
rmse_test = []
r2_scores = []

for i in range(1, 11):
    X_train, X_test, Y_train, Y_test = train_test_split(SS_X, Y, test_size=0.30, random_state=i)
    
    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    # Prediction on the test set
    Y_pred_test = model.predict(X_test)
    
    # Model Evaluation
    rmse_test.append(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))
    r2_scores.append(r2_score(Y_test, Y_pred_test))

# Display RMSE and R-squared values
print("Average RMSE of test set:", np.mean(rmse_test).round(3))
print("Average R-squared on test set:", np.mean(r2_scores).round(3))

# Model Deletion Diagnostics==================================================
model = sm.OLS(Y, sm.add_constant(SS_X)).fit()

# Cook's Distance
influence = model.get_influence()
cook_distance = influence.cooks_distance
cook_threshold = 4 / len(df)

# Leverage Cutoff
leverage = influence.hat_matrix_diag
leverage_threshold = 2 * (SS_X.shape[1] + 1) / len(df)

# Identify influential points=================================================
influential_points = np.where((cook_distance > cook_threshold) | (leverage > leverage_threshold))[0]

cook_distance = np.array(influence.cooks_distance[0])
leverage = np.array(influence.hat_matrix_diag)
influential_points = np.where((cook_distance > cook_threshold) | (leverage > leverage_threshold))[0]


# Remove influential points====================================================
df_cleaned = df.drop(index=influential_points)

# One-hot encode the 'State' column
X_cleaned_encoded = pd.get_dummies(df_cleaned['State'], drop_first=True)
X_cleaned_encoded = pd.concat([df_cleaned[['R&D Spend', 'Administration', 'Marketing Spend']], X_cleaned_encoded], axis=1)

# Separate features and target variable
X_cleaned = X_cleaned_encoded
Y_cleaned = df_cleaned['Profit']




# Standardization using the same scaler instance
X_cleaned_standardized = SS.fit_transform(X_cleaned)


final_model = sm.OLS(Y_cleaned, sm.add_constant(X_cleaned_standardized)).fit()

# Display final R-squared values
print("\nFinal R-squared:", final_model.rsquared)
print("Final Adjusted R-squared:", final_model.rsquared_adj)
