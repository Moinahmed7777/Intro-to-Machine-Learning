# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:03:43 2021

@author: Necro
"""

import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/iris.csv")
#EDA
print(df.head())
print(df.info())

print(df.head())
print(df.describe())

X = df.drop("species",axis=1)
y =df["species"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = RandomForestClassifier(n_estimators=10, max_features="auto",random_state=101 )

model.fit(X_train,y_train)

preds = model.predict(X_test)

print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))
print(pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=["Feature Importance"]))
