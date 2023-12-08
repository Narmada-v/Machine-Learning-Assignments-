# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:03:19 2023

@author: narma
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset=============================================================
toyo = pd.read_csv('C:/Users/narma/Downloads/ToyotaCorolla.csv', encoding='latin1')

# Select relevant columns for the model
toyo_subset = toyo[['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight']]

# Check for missing values======================================================
toyo_subset.info()

# Handle missing values if any
toyo_subset = toyo_subset.dropna()

# Correlation analysis=========================================================
correlation_matrix = toyo_subset.corr()

# Multicollinearity check (VIF)================================================
numeric_features = toyo_subset.drop('Price', axis=1)
vif_data = pd.DataFrame()
vif_data["Variable"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(numeric_features.shape[1])]
print("VIF:")
print(vif_data)

# Model fitting===============================================================
formula = 'Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight'
model = smf.ols(formula, data=toyo_subset).fit()
print(model.summary())

# Residual analysis
# Plotting residuals vs. predicted values======================================
sns.scatterplot(x=model.fittedvalues, y=model.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

# Leverage vs. Residuals squared plot==========================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.residplot(x=model.get_influence().hat_matrix_diag, y=model.resid, lowess=True, scatter_kws={'alpha': 0.8}, line_kws={'color': 'red', 'alpha': 0.5})
plt.xlabel('Leverage')
plt.ylabel('Standardized Residuals')
plt.title('Leverage vs. Standardized Residuals')
plt.show()

# Cook's distance==============================================================
influence = model.get_influence()
cook_distance = influence.cooks_distance[0]


# Identify influential points using Cook's distance
cook_threshold = 4 / len(toyo_subset)
influential_points = np.where(cook_distance > cook_threshold)[0]
print("Influential Points:", influential_points)

# Model deletion techniques
# Leverage Cutoff===============================================================
leverage_cutoff = 2 * (toyo_subset.shape[1] + 1) / len(toyo_subset)

# Identify high leverage points================================================
high_leverage_points = np.where(model.get_influence().hat_matrix_diag > leverage_cutoff)[0]
print("High Leverage Points:", high_leverage_points)

# Remove influential points and refit the model================================
toyo_cleaned = toyo_subset.drop(index=influential_points)
model_cleaned = smf.ols(formula, data=toyo_cleaned).fit()

# Display final R-squared values===============================================
print("\nFinal R-squared:", model_cleaned.rsquared)
print("Final Adjusted R-squared:", model_cleaned.rsquared_adj)
