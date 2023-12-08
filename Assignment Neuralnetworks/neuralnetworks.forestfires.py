# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:32:37 2023

@author: narma
"""

import pandas as pd
import numpy as np
import sklearn
df = pd.read_csv("C:/Users/narma/Downloads/Neuralnetworks/gas_turbines.csv")
df.head()
df.describe()
df.shape
df.info()
df.columns
df.isnull().sum()
# check for duplicate data
duplicate = df.duplicated()
print(duplicate.sum())
df[duplicate]

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

for i in df.columns:
    print(i)
    sns.boxplot(df[i],color = 'green')
    plt.show()
    
# Boxplot of Turbine Energy Yield (TEY)
plt.boxplot(df['TEY'])

sns.boxplot(df['TEY'], color = 'green')

X = df.loc[:,['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO','NOX']]
y= df.loc[:,['TEY']]

!pip install tensorflow
import tensorflow as tf
print(tf.__version__)


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model




import keras
print(keras.__version__)

!pip install scikeras
import keras
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import InputLayer,Dense
estimator = KerasRegressor(build_fn=baseline_model,epochs=50, batch_size=100, verbose=False)
kfold = KFold(n_splits=5)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, y)
prediction = estimator.predict(X)
prediction

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
estimator.fit(X_train, y_train)
prediction = estimator.predict(X_test)
prediction

X = df.drop(columns = ['TEY'], axis = 1) 
y = df.iloc[:,7]
from sklearn.preprocessing import scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)
X_test_scaled

import tensorflow as tf
input_size = len(X.columns)
output_size = 1
hidden_layer_size = 50

model = tf.keras.Sequential([
                                
                               tf.keras.layers.Dense(hidden_layer_size, input_dim = input_size, activation = 'relu'),
                               tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                               tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                               tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),     
                               tf.keras.layers.Dense(output_size)
                             ])

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.03)
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['MeanSquaredError'])
num_epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)
model.fit(X_train_scaled, y_train, callbacks = early_stopping, validation_split = 0.1, epochs = num_epochs, verbose = 2)

test_loss, mean_squared_error = model.evaluate(X_test_scaled, y_test)

predictions = model.predict_on_batch(X_test_scaled)
plt.scatter(y_test, predictions)

predictions_df = pd.DataFrame()
predictions_df['Actual'] = y_test
predictions_df['Predicted'] = predictions
predictions_df['% Error'] = abs(predictions_df['Actual'] - predictions_df['Predicted'])/predictions_df['Actual']*100
predictions_df.reset_index(drop = True)