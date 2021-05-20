# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 03:33:55 2021

@author: Necro
"""
import pandas as pd 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report


from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/bank_note_data.csv")

print(df.head())
C1= df["Image.Curt"]
X = df.drop(["Class"],axis=1)
y = df["Class"]
print(X.head())

print(df.head())
print(df.info())

print(df.head())
print(df.describe())


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state = 42)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

sns.pairplot(df, hue="Class",corner=True)

model = RandomForestClassifier(n_estimators=25, max_features="auto",random_state=101 )
model.fit(scaled_X_train,y_train)

base_pred =model.predict(scaled_X_test)
print(confusion_matrix(y_test,base_pred))
print(classification_report(y_test,base_pred))
print(pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=["Feature Importance"]))



