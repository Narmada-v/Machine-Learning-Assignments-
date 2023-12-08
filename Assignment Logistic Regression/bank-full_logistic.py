# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:51:40 2023

@author: narma
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import  metrics
from sklearn.metrics import classification_report

df= pd.read_csv('C:/Users/narma/Downloads/bank-full.csv',delimiter=';',quotechar='"')
df.to_csv('new_data.csv', index=False)  # Save to a new CSV file without including the index
df

df.info()
#gives null values present in columns
df.isnull().sum()
df.describe
# One-Hot Encoding of categrical variables====================================================
df1=pd.get_dummies(df,columns=['job','marital','education','contact','poutcome'])
df1

# To see all columns
pd.set_option("display.max.columns", None)
df1

# Custom Binary Encoding of Binary o/p variables============================================== 
df1['default'] = np.where(df1['default'].str.contains("yes"), 1, 0)
df1['housing'] = np.where(df1['housing'].str.contains("yes"), 1, 0)
df1['loan'] = np.where(df1['loan'].str.contains("yes"), 1, 0)
df1['y'] = np.where(df1['y'].str.contains("yes"), 1, 0)
df1

df1['month'].value_counts()

order={'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}
df1=df1.replace(order)
df1

df1.info()

from sklearn.preprocessing import LabelEncoder
# Initialize the LabelEncoder=================================================================
label_encoder = LabelEncoder()

# Columns to label encode
boolean_columns = ['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
                    'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician',
                    'job_unemployed', 'job_unknown', 'marital_divorced', 'marital_married', 'marital_single',
                    'education_primary', 'education_secondary', 'education_tertiary', 'education_unknown',
                    'contact_cellular', 'contact_telephone', 'contact_unknown', 'poutcome_failure',
                    'poutcome_other', 'poutcome_success', 'poutcome_unknown']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to boolean columns===================================================
df1[boolean_columns] = df1[boolean_columns].apply(lambda col: label_encoder.fit_transform(col))

# Display the updated DataFrame
print(df1.info())

Y = df1['y']

df1
# Drop the target variable from the DataFrame to get the feature matrix X
X = df1.drop('y', axis=1)

# Display the feature matrix X
print(X.head())

#Standardization======================================================================================
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
pd.DataFrame(SS_X)
#=================================================================================================
# step5: Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30)

'''
print("X_train data size: ",X_train.shape)
print("X_test data size: ",X_test.shape)

print("Y_train data size: ",Y_train.shape)
print("Y_test data size: ",Y_test.shape)
'''

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(SS_X,Y)
Y_pred = logreg.predict(SS_X)
#================================================================================================
# step6: Metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y,Y_pred)
cm
print("Accuacy score:", accuracy_score(Y,Y_pred).round(2))
logreg.predict(X)
logreg.predict_proba(SS_X) # 1- prob, prob
Y_probabilities = logreg.predict_proba(SS_X)


#ROC CURVE=======================================================================================
from sklearn.metrics import roc_curve,roc_auc_score
fpr, tpr, dummy = roc_curve(Y.values.ravel(), Y_probabilities[:, 1])
fpr, tpr, dummy = roc_curve(Y, Y_probabilities[:, 1])


import matplotlib.pyplot as plt
plt.scatter(x = fpr,y=tpr)
plt.plot(fpr,tpr,color='red')
plt.ylabel("True positive Rate")
plt.xlabel("False positive Rate")
plt.show()

print("AUC score:", roc_auc_score(Y.values.ravel(), Y_probabilities[:, 1]).round(3))
