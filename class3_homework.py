#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:17:26 2021

@author: priyanshu
"""

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('CovidHW.csv')

dataset.plot(x='Tested', y='Positive' , style='o')
plt.title('Tested vs Positive')
plt.xlabel('Tested')
plt.ylabel('Positive')
plt.show()

X = dataset['Tested'].values.reshape(-1,1)
y = dataset['Positive'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

#To retrieve the intercept
print(regressor.intercept_)
#To retrive the slope
print(regressor.coef_)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual' : y_test.flatten(), 'Predicted':y_pred.flatten()})
print(df)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='red', linewidth=1)
plt.show()

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("Fitting score: ", r2)