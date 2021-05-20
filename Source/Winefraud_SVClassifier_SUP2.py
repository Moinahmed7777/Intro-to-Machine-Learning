# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:12:15 2021

@author: Necro
"""
#1.	Import libraries for df and plotting
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pd.set_option("display.max_rows", None, "display.max_columns", None)
#2.	Use pandas to read wine_fraud.csv
df = pd.read_csv("../Datasets/wine_fraud.csv")
#3.	Check out info(), head(), and describe methods 
print(df.head())
print(df.info())

print(df.describe())
#4.	What are the unique variables in the target column we are trying to predict (quality)?
print(df["quality"].unique())
#5.	Create a countplot that displays the count per category of Legit vs Fraud.
print(df["quality"].value_counts())
#sns.countplot(data=df,x="quality")
#6
sns.countplot(data=df,x="type",hue="quality")


#print("#####################################")
#7 & 9 (didn't not map since get_dummies() does it and is asked later)
X = pd.get_dummies(df)
print(X.head(20))
correlation = X.corr()
#8
print(correlation)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(24,12))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap="viridis",ax=ax1)
#sns.barplot(x=correlation,corner=True,data=X,ax=ax2)
sns.pairplot(df, hue="quality",corner=True)
########

#10
X.drop(["quality_Legit"],axis=1,inplace=True)
y= X["quality_Fraud"]
X.drop(["quality_Fraud"],axis=1,inplace=True)
X.info()

#11
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#12
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#13
#SVC defaults, C=1.0, kernel ='rbf', gamma = 'scale',
basemod = SVC(class_weight="balanced")
basemod.fit(scaled_X_train, y_train)

basepreds=basemod.predict(scaled_X_test)

print(confusion_matrix(y_test,basepreds))
print(classification_report(y_test,basepreds))

#14
#GridSearch for auto-find* better model 
param_grid = {'C':[0.001,0.01,0.1,0.5,1],'gamma':['scale','auto'], 'kernel':['linear','rbf']}


grid = GridSearchCV(SVC(),param_grid)#,refit = True, verbose=2)

grid.fit(scaled_X_train,y_train)

print(grid.best_params_)
##OUTPUT##->{'C': 0.001, 'gamma': 'scale', 'kernel': 'linear'}

basemod2 = SVC(C=0.001,kernel ='linear',class_weight="balanced")
basemod2.fit(scaled_X_train, y_train)

basepreds2=basemod2.predict(scaled_X_test)
#15
print(confusion_matrix(y_test,basepreds2))
print(classification_report(y_test,basepreds2))




#17,18
#DecisionTree
model =  DecisionTreeClassifier()
model.fit(scaled_X_train,y_train)

base_pred =model.predict(scaled_X_test)
print(confusion_matrix(y_test,base_pred))
print(classification_report(y_test,base_pred))

