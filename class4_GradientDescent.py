#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:38:59 2021

@author: priyanshu
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

m_curr = float(0)
b_curr = float(0)

lr = 0.1

n = len(x)

itr = 100 

plt.scatter(x,y)

cost = []

for i in range(itr):
    y_pred = m_curr*x + b_curr
    cost_tmp = (1/n)*sum([val**2 for val in (y - y_pred)])
    
    cost.append(cost_tmp)
    
    dm = -(2/n)*sum(x*(y - y_pred)) #gradient
    db = -(2/n)*sum(y - y_pred)     #gradient
    m_curr = m_curr - lr*dm
    b_curr = b_curr - lr*db
    
    print("m {}, b {}, cost {}; iteration {}".format(m_curr, b_curr, cost_tmp, i))
    plt.plot(x , y_pred)
    
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print(""r2)
plt.show()
plt.figure
index = np.arange(100)
plt.scatter(index,cost)
plt.show()



