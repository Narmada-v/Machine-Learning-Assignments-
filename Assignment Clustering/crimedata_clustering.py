# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:35:42 2023

@author: narma
"""
#IMPORTING====================================================
import pandas as pd
df = pd.read_csv("C:/Users/narma/Downloads/crime_data (1).csv")
df.shape
df.head()
list(df)


X=df.iloc[:,1:]
X
#DENDOGRAM==========================================================
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("Customer Dendrogram")
dend=shc.dendrogram(shc.linkage(X,method='average'))

#AGGLOMERATIVE CLUSTERING======================================
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='complete')
Y=cluster.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()
Y


#KMEANS======================================================
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

#DBSCAN==============================================================
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
SS_X=SS.fit_transform(X)
pd.DataFrame(SS_X)

X=df.iloc[:,1:]
X
from sklearn.cluster import DBSCAN
DBSCAN()
dbscan=DBSCAN(eps=0.75,min_samples=3)
dbscan.fit(SS_X)

#NOISY SAMPLES ARE REMOVED==============================================
dbscan.labels_
c1=pd.DataFrame(dbscan.labels_,columns=['cluster'])
print(c1['cluster'].value_counts())

clustered=pd.concat([df,c1],axis=1)
noisedata=clustered[clustered['cluster']==-1]
finaldata=clustered[clustered['cluster']==0]
finaldata
#========================================================================
from sklearn.cluster import KMeans
KMeans=KMeans(n_clusters=5,n_init=30)
KMeans.fit(finaldata.iloc[:,1:])
Y=KMeans.predict(finaldata.iloc[:,1:])
Y=pd.DataFrame(Y)
Y[0].value_counts()