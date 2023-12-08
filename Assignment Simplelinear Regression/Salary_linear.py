# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:34:34 2023

@author: narma
"""

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load dataset
dataset = pd.read_csv('C:/Users/narma/Downloads/Salary_Data.csv')

# Check dataset information
dataset.info()

# Explore the distribution of variables
sns.distplot(dataset['Salary'])
sns.distplot(dataset['YearsExperience'])

# Check correlation between variables
correlation_matrix = dataset.corr()
print(correlation_matrix)

# Original Simple Linear Regression Model
model = smf.ols("Salary ~ YearsExperience", data=dataset).fit()

# Display original model parameters
print("Original Model Parameters:")
print(model.params)

# Display original R-squared values
print("\nOriginal R-squared:", model.rsquared)
print("Original Adjusted R-squared:", model.rsquared_adj)

# Manual prediction for say sorting time 5 in the original model
salary_original = model.predict(pd.DataFrame({'YearsExperience': [5]}))
print("\nOriginal Model - Manual Prediction for YearsExperience = 5:")
print(salary_original.iloc[0])

# Transformational Models
transformations = ['log', 'sqrt', 'square']

for transformation in transformations:
    # Apply transformation to the predictor variable
    dataset['YearsExperience_' + transformation] = getattr(np, transformation)(dataset['YearsExperience'])

    # Build the transformed model
    transformed_model = smf.ols(f"Salary ~ YearsExperience_{transformation}", data=dataset).fit()

    # Display transformed model parameters
    print(f"\n{transformation.capitalize()} Transformed Model Parameters:")
    print(transformed_model.params)

    # Display transformed R-squared values
    print(f"{transformation.capitalize()} Transformed R-squared:", transformed_model.rsquared)
    print(f"{transformation.capitalize()} Transformed Adjusted R-squared:", transformed_model.rsquared_adj)

    # Manual prediction for say sorting time 5 in the transformed model
    salary_transformed = transformed_model.predict(pd.DataFrame({'YearsExperience_' + transformation: [5]}))
    print(f"{transformation.capitalize()} Transformed Model - Manual Prediction for YearsExperience = 5:")
    print(salary_transformed.iloc[0])

    #RMSE for the transformed model
    predictions_transformed = transformed_model.predict(dataset)
    rmse_transformed = sqrt(mean_squared_error(dataset['Salary'], predictions_transformed))
    print(f"{transformation.capitalize()} Transformed Model - RMSE:", rmse_transformed)
