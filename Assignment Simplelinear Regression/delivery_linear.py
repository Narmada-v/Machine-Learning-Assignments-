# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:22:58 2023

@author: narma
"""

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from math import sqrt

# Import dataset
dataset = pd.read_csv('C:/Users/narma/Downloads/delivery_time.csv')

# Display dataset information
dataset.info()

# Plot distributions
sns.distplot(dataset['Delivery Time'])
sns.distplot(dataset['Sorting Time'])

# Renaming Columns
dataset = dataset.rename({'Delivery Time': 'delivery_time', 'Sorting Time': 'sorting_time'}, axis=1)

# Correlation matrix
dataset.corr()

# Simple Linear Regression Model
model = smf.ols("delivery_time ~ sorting_time", data=dataset).fit()

# Finding Coefficient parameters
model.params

# Finding tvalues and pvalues
model.tvalues, model.pvalues

# Finding Rsquared Values
model.rsquared, model.rsquared_adj

# Manual prediction for say sorting time 5
delivery_time = model.params.Intercept + model.params.sorting_time * 5
delivery_time

# Automatic Prediction for say sorting time 5, 8
new_data = pd.Series([5, 8], name='sorting_time')

# Transformations
transformations = {'Original': new_data,
                   'Log': np.log(new_data),
                   'Square': np.square(new_data),
                   'Sqrt': np.sqrt(new_data)}

# Create a DataFrame to store predictions for each transformation
predictions_df = pd.DataFrame(transformations)
predictions_df


# Display predictions and RMSE values
print(predictions_df)
