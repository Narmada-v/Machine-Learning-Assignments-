# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:05:21 2023

@author: narma
"""
#import the dataset
import pandas as pd
df = pd.read_csv('C:/Users/narma/Downloads/Association rules/my_movies.csv')
df
df.shape
df.head()
list(df)


# Remove the first 5 columns===========================================================
df_new = df.iloc[:, 5:]

#new CSV file
df_new.to_csv('updated_dataset.csv', index=False)
df_new

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#different support and confidence values==============================================
support_values = [0.1, 0.15, 0.2]  
confidence_values = [0.6, 0.7, 0.8]  


for support in support_values:
    for confidence in confidence_values:
        frequent_itemsets = apriori(df_new, min_support=support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        print(f"Support: {support}, Confidence: {confidence}, Number of rules: {len(rules)}")
        
# Try different minimum length values===================================================
min_length_values = [2, 3, 4]

for min_length in min_length_values:
    frequent_itemsets = apriori(df_new, min_support=0.1, use_colnames=True, max_len=min_length)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    print(f"Minimum Length: {min_length}, Number of rules: {len(rules)}")

#matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()

#bar chart for top 10 rules based on lift=============================================
top_rules = rules.nlargest(10, 'lift') 
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_rules)), top_rules['lift'], align='center')
plt.yticks(range(len(top_rules)), top_rules['antecedents'] + ' -> ' + top_rules['consequents'])
plt.xlabel('Lift')
plt.title('Top 10 Rules by Lift')
plt.show()

# Visualize rules using a network graph================================================
import networkx as nx

G = nx.DiGraph()
for i in range(len(rules)):
    G.add_edge(rules.iloc[i]['antecedents'], rules.iloc[i]['consequents'], weight=rules.iloc[i]['lift'])

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('Association Rules Network Graph')
plt.show()

        
        
        
        
