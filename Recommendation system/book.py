# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:02:39 2023

@author: narma
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/narma/Downloads/book.csv", encoding='latin1')
df
df.head()
df.tail()
df.shape
df.info()
df.isnull().sum()
df.drop(df.columns[0],axis=1,inplace=True)
df

# Renaming the columns
df.columns = ["User_ID","Book_Title","Book_Rating"]
df
df = df.sort_values(by=['User_ID'])
df
df.nunique()
df.loc[df["Book_Rating"] == 'small', 'Book_Rating'] = 0
df.loc[df["Book_Rating"] == 'large', 'Book_Rating'] = 1

df.Book_Rating.value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,6))
sns.histplot(df.Book_Rating)

book_df = df.pivot_table(index='User_ID',
                   columns='Book_Title',
                   values='Book_Rating').reset_index(drop=True)


book_df.fillna(0,inplace=True)

book_df

# Average rating of books
avg = df['Book_Rating'].mean()


avg

# Calculate the minimum number of votes required to be in the chart, 
minimum = df['Book_Rating'].quantile(0.90)
minimum

# Filter out all qualified Books into a new DataFrame
q_Books = df.copy().loc[df['Book_Rating'] >= minimum]
q_Books.shape


# # Calculating Cosine Similarity between Users


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


user_sim=1-pairwise_distances(book_df.values,metric='cosine')


user_sim

user_sim_df=pd.DataFrame(user_sim)

user_sim_df

#Set the index and column names to user ids 
user_sim_df.index = df.User_ID.unique()
user_sim_df.columns = df.User_ID.unique()

user_sim_df


np.fill_diagonal(user_sim,0)
user_sim_df

#Most Similar Users
print(user_sim_df.idxmax(axis=1))
print(user_sim_df.max(axis=1).sort_values(ascending=False).head(50))

reader = df[(df['User_ID']==1348) | (df['User_ID']==2576)]
reader


reader1=df[(df['User_ID']==1348)] 

reader1

reader2=df[(df['User_ID']==2576)] 

reader2
