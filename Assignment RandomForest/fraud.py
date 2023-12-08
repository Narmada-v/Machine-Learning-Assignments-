# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:22:39 2023

@author: narma
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# IMPORT TYHE CSV FILE===================================================================
df = pd.read_csv('C:/Users/narma/Downloads/random forest/Fraud_check.csv')
df
df.head()
df.info()
list(df)
df.describe()

#EDA===============================================================
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

columns_to_plot = ["Taxable.Income", "Work.Experience", "City.Population"]
titles = ["Taxable Income", "Work.Experience", "City Population"]

for i, column in enumerate(columns_to_plot):
    axs[i].boxplot(df[column])
    axs[i].set_title(titles[i])

plt.tight_layout()
plt.show()

sns.pairplot(df)

#Taxable . Income categorization===================================================
Conditions=[(df["Taxable.Income"]<=30000),(df["Taxable.Income"]>30000)]
Categories=["Risky","Good"]

df["Taxable.Income"]=np.select(Conditions,Categories)
df["Taxable.Income"].unique()
df["Taxable.Income"].describe()
df["Taxable.Income"].info()
df
list(df)

#Label Encoding===================================================================
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df["Taxable.Income"]=label.fit_transform(df["Taxable.Income"])
df=pd.get_dummies(df,columns=["Undergrad","Marital.Status","Urban"])
df


X=df.drop(["Taxable.Income"],axis=1)
Y=df["Taxable.Income"]
plt.title("Class 0 and 1 distribution",color="blue")
plt.pie(Y.value_counts(), labels=Y.unique(), shadow=True, autopct='%1.1f%%')
plt.show()

#train test split=================================================================
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0,shuffle=True)
X.describe()

Y.describe()
Y[Y==1].value_counts()
Y[Y==0].value_counts()
df[df.duplicated()]


#Random Forest========================================================================
est=RandomForestClassifier(max_depth=5,min_samples_split=2,random_state=42)
params = {"criterion":('gini','entropy'),"n_estimators":(1,100,5),"max_features": ("auto", "sqrt", "log2")
         }
GCV=GridSearchCV(estimator=est,param_grid=params,cv=5)

modelgcv=GCV.fit(X_train,Y_train)
best_params = GCV.best_params_
best_score = GCV.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)


# Final model with best parameters====================================================
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, Y_train)

# Predictions on the test set
predictions = final_model.predict(X_test)

# Evaluating the final model
accuracy = accuracy_score(Y_test, predictions)
report = classification_report(Y_test, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)