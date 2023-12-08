# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:27:38 2023

@author: narma
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

#import the dataset
df = pd.read_csv('C:/Users/narma/Downloads/Association rules/book.csv')
df

#Apriori algorithm with different support and confidence values
support_values = [0.1, 0.15, 0.2]  
confidence_values = [0.5, 0.6, 0.7] 
for support in support_values:
    for confidence in confidence_values:
        frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        print(f"Support: {support}, Confidence: {confidence}, Number of Rules: {len(rules)}")

# Change the minimum length in Apriori algorithm
min_length_values = [2, 3, 4]

for length in min_length_values:
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    filtered_rules = rules[rules['antecedents'].apply(lambda x: len(x) >= length)]
    print(f"Minimum Length: {length}, Number of Rules: {len(filtered_rules)}")

# Visualize the obtained rules using different plots
# For example, you can visualize support vs. confidence using a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs. Confidence')
plt.show()


