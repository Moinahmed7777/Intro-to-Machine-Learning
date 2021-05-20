# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:11:12 2021

@author: Necro
"""


import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/penguins_size.csv")

#Exploratory Data Analysis

print(df.head())


df2 = df #.drop(["species"],axis=1)

print(df2.head())

print(df2.info())

df2.dropna(inplace=True)

print(df2.info())

X = pd.get_dummies(df2.drop('species',axis=1),drop_first=True)
#X2= X.drop(["sex_MALE"],axis=1)
#print(X2.info())
print(X.head())
scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)



model = KMeans(n_clusters=5)
cluster_labels = model.fit_predict(scaled_X)

print(cluster_labels)


ssd = []
for i in range(2,10):
    model = KMeans(n_clusters=i)
    model.fit(scaled_X)
    #sum of sqaured distances of samples to their closest cluster center
    ssd.append(model.inertia_)


plt.plot(range(2,10),ssd,'o--')
plt.xlabel("K Value")
plt.ylabel(" Sum of squared distances")
plt.show()
