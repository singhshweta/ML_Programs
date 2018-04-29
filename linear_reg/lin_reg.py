# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 00:07:07 2018

@author: shweta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



train_set = pd.read_csv("train.csv")
train_set[['x', 'y']] = train_set[['x','y']].fillna(value=0)

df_set = train_set.copy()

sns.regplot("x", "y",data=df_set)

train_set = train_set.drop(["y"], axis=1)
train_labels = df_set["y"]
sns.regplot(train_set, train_labels)

plt.show()
plt.scatter(train_set,train_labels)
plt.show()

test_set = pd.read_csv("test.csv")

test_set[['x', 'y']] = test_set[['x','y']].fillna(value=0)
test_set_full = test_set.copy()
test_set = test_set.drop(["y"], axis=1)
lin_reg = LinearRegression()

lin_reg.fit(train_set, train_labels)

print("Coefficients: ", lin_reg.coef_)
print("Intercept: ", lin_reg.intercept_)

salary_pred = lin_reg.predict(test_set)
accuracy = lin_reg.score(test_set, test_set_full["y"])
print(accuracy)

