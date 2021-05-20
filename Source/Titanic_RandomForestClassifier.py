# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:24:21 2021

@author: Necro
"""


import pandas as pd 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/titanic.csv")
#EDA
#df.dropna()
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

print(sns.pairplot(df, hue="survived",corner=True))
y = df["survived"]
X.drop(['survived'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = RandomForestClassifier(n_estimators=25, max_features="auto",random_state=101 )

model.fit(X_train,y_train)

preds = model.predict(X_test)

print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))
print(pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=["Feature Importance"]))