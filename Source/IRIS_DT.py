# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:03:43 2021

@author: Necro
"""

import pandas as pd 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/iris.csv")
#EDA
print(df.head())
print(df.info())

print(df.head())
print(df.describe())
print(sns.pairplot(df, hue="species",corner=True))
X = df.drop("species",axis=1)
y =df["species"]

#print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model =  DecisionTreeClassifier()
model.fit(X_train,y_train)

base_pred =model.predict(X_test)
print(confusion_matrix(y_test,base_pred))
print(classification_report(y_test,base_pred))