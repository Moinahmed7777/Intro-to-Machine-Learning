# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:03:09 2021

@author: Necro
"""

import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv("../Datasets/loan_data.csv")
#display all*more rows
pd.set_option("display.max_rows", None, "display.max_columns", None)
# 3. Chekc out info(), head(), and desribe methods 
#EDA
print(df.info())

print(df.head())
print(df.describe())

# 4. Create a plot for each credit.policy (both plots in the same figure)
#sns.pairplot(df, hue="credit.policy",corner=True)


# 5. Create a plot for each credit.policy, but select by the not.fully.paid column
#sns.pairplot(df, hue="not.fully.paid",corner=True)

# 6. Create a countplot for the counts of loan purpose, pick a hue defined by not.fully.paid
# 7. Create a joint plot between FICO score and interest rate 
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(24,12))
sns.countplot(data=df,x="purpose",hue="not.fully.paid",ax=ax1)
sns.jointplot(data=df, x="fico", y="int.rate",ax=ax2)

cut = pd.get_dummies(df)


X = pd.get_dummies(df.drop('purpose',axis=1),drop_first=True)
X.drop('credit.policy',axis=1,inplace=True)
y = df["credit.policy"]
print(X.tail())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

base_pred =model.predict(X_test)
print(confusion_matrix(y_test,base_pred))
print(classification_report(y_test,base_pred))