# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:03:40 2021

@author: Necro
"""

#import libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

#print all rows
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/titanic.csv")
#df.dropna()
#EDA
print(df.head(20))
print(df.info())

print(df.head())

df.drop(['cabin','name','boat','body','home.dest'], axis=1, inplace=True)
df.dropna(inplace=True)
print(df.head(20))

print(len(df["ticket"].unique()))
cut=pd.get_dummies(df)
X = pd.get_dummies(df.drop('ticket',axis=1),drop_first=True)
print(X .head(20))
print(X.info())
y = df["survived"]
X.drop(['survived'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

base_pred =model.predict(X_test)
print(confusion_matrix(y_test,base_pred))
print(classification_report(y_test,base_pred))