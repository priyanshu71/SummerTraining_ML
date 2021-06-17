#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:10:11 2021

@author: priyanshu
"""

import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

train = pd.read_csv('/home/priyanshu/Documents/Python/Datasets/train.csv')
train_dataset = train

#data processing and cleaning
train_dataset['Age'].fillna(train_dataset['Age'].median(), inplace=True)
train_dataset["Sex"][train_dataset["Sex"]=="male"] = 0
train_dataset["Sex"][train_dataset["Sex"]=="female"] = 1


X = train_dataset[['Pclass', 'Sex', 'Age','SibSp', 'Parch']]
y = train_dataset['Survived']

X = sm.add_constant(X)

# y = b0 + b1*tmp1 + b2*tmp2 + b3*tmp3 + b4*tmp4 + b5*tmp5

#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 123)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("Fitting score: ", r2)

numeric = train_dataset.columns[[1,2,3,4,5]]

plt.figure(figsize=(15,12))
for i in range(1,5):
    plt.subplot(4, 1, i)
    plt.scatter(train_dataset['Survived'], train_dataset[numeric[i-1]], s = 1)
    plt.xlabel('Survived')
    plt.ylabel(numeric[i-1])










"""
tmp1 = X['Pclass']
tmp2 = X['Sex']
tmp3 = X['Age']
tmp4 = X['SibSp']
tmp5 = X['Parch']

np.random.seed(0)
theta=np.random.randn(1,5)
print(theta)


iterations = 1000
n = np.size(tmp1)
lr = 0.01 


b_c, b1_c, b2_c, b3_c, b4_c, b5_c = float(0), float(0), float(0), float(0), float(0), float(0)


cost = []
for i in range(iterations):
    y_pred = b_c + b1_c*tmp1 + b2_c*tmp2 + b3_c*tmp3 + b4_c*tmp4 + b5_c*tmp5
    
    cost_tmp = (1/n)*sum([val**2 for val in (y - y_pred)])
    cost.append(cost_tmp)
    
    db_c  = 
    db1_c =
    db2_c =
    db3_c =
    db4_c =
    db5_c =
    
"""    

