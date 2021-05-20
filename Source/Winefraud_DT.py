# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 06:58:50 2021

@author: Necro
"""


import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
import matplotlib.pyplot as plt



pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/wine_fraud.csv")
print(df.head())

X = df.drop(["quality"],axis=1)
y = df["quality"]


X2 = pd.get_dummies(X)
print(X2.head())
print(y.head())



X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

model =  DecisionTreeClassifier()
model.fit(scaled_X_train,y_train)

base_pred =model.predict(scaled_X_test)
print(confusion_matrix(y_test,base_pred))
print(classification_report(y_test,base_pred))