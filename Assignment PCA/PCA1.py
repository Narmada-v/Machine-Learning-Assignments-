# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:51:04 2023

@author: narma
"""

import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


#importing the file=============================================================
df = pd.read_csv("C:/Users/narma/Downloads/wine.csv")
df.shape
df.head()
list(df)

X = df.iloc[:,1:]
# standardization==============================================================
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
pd.DataFrame(SS_X)

# Perform PCA==================================================================
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(SS_X)
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])


# K-means clustering===========================================================
inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pc_df)
    inertia.append(kmeans.inertia_)

# Plot the scree plot or elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Scree Plot / Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Perform hierarchical clustering==============================================
linkage_matrix = linkage(pc_df, method='ward')

# Plot the dendrogram
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Choose the optimal number of clusters based on the scree plot or dendrogram
# For K-means, it's the "elbow" point; for hierarchical, it's where the dendrogram cuts horizontally

# Example: Suppose you find that 3 clusters are optimal

# K-means with optimal number of clusters======================================

from sklearn.cluster import KMeans
KMeans=KMeans(n_clusters=3,n_init=30)
KMeans.fit(X)
Y=KMeans.predict(X)
Y=pd.DataFrame(Y)
Y[0].value_counts()
Y


KMeans.inertia_

inertia=[]
from sklearn.cluster import KMeans
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)
    
    
plt.scatter(range(1,11),inertia)
plt.plot(range(1,11),inertia,color='red')
plt.title('ELBOW METHOD')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()


