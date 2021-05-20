# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:12:15 2021

@author: Necro
"""

import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/wine_fraud.csv")

print(df.head())
print(df.info())

print(df.describe())

print(df["quality"].unique())

print(df["quality"].value_counts())
#sns.countplot(data=df,x="quality")
sns.countplot(data=df,x="type",hue="quality")

print("#####################################")
X = pd.get_dummies(df)
print(X.head(20))
correlation = X.corr()

print(correlation)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(24,12))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap="viridis",ax=ax1)
#sns.barplot(x=correlation,corner=True,data=X,ax=ax2)
sns.pairplot(df, hue="quality",corner=True)
########

X.drop(["quality_Legit"],axis=1,inplace=True)
y= X["quality_Fraud"]
X.drop(["quality_Fraud"],axis=1,inplace=True)
X.info()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#SVC defaults, C=1.0, kernel ='rbf', gamma = 'scale',
basemod = SVC(class_weight="balanced")
basemod.fit(scaled_X_train, y_train)

basepreds=basemod.predict(scaled_X_test)

print(confusion_matrix(y_test,basepreds))
print(classification_report(y_test,basepreds))


#GridSearch for auto-find* better model 
param_grid = {'C':[0.001,0.01,0.1,0.5,1],'gamma':['scale','auto'], 'kernel':['linear','rbf']}


grid = GridSearchCV(SVC(),param_grid)#,refit = True, verbose=2)

grid.fit(scaled_X_train,y_train)

print(grid.best_params_)
##OUTPUT##->{'C': 0.001, 'gamma': 'scale', 'kernel': 'linear'}

basemod2 = SVC(C=0.001,kernel ='linear',class_weight="balanced")
basemod2.fit(scaled_X_train, y_train)

basepreds2=basemod2.predict(scaled_X_test)

print(confusion_matrix(y_test,basepreds2))
print(classification_report(y_test,basepreds2))


