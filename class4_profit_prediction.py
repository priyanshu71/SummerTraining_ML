#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:14:10 2021

@author: priyanshu
"""


import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder 
import matplotlib.pyplot as plt

companies = pd.read_csv('/home/priyanshu/Documents/Python/Datasets/1000_Companies.csv')
data = companies
companies.head()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()

data['State'] = le.fit_transform(data['State'])

columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')

data = np.array(columnTransformer.fit_transform(data), dtype = np.float64)

#extracting features
X = data[:,:-1] #all rows, all colums except -1 (last column) 
#extracting targets
y = data[:,-1] #all rows, last column

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print(lin_reg.coef_)
print(lin_reg.intercept_)

from sklearn.metrics import r2_score
r2 = r2_score(y_pred, y_test)
print('prediction accuracy:', r2)