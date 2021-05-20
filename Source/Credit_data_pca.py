# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:15:27 2021

@author: Necro
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/credit_data.csv")

print(df.head())

scaler = StandardScaler()
 

scaler.fit(df)
scaled_data = scaler.transform(df)
print(scaled_data[0])

#PCA PC1,PC2

pca = PCA(n_components=2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c=df['default'], cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')


print(pca.components_)

#feature_names = list(X.columns)
df_comp = pd.DataFrame(pca.components_, columns=df.columns)
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')



#NN 
features = pca.components_

#print(features)

#print(type(df.default))
#series to numpy array
y = np.array(df.default).reshape(-1, 1)

#print(type(y))

# we have 2 classes(target) so the labels will have 2 values
# first class: (1,0) second class: (0,1)
encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(x_pca, targets, test_size=0.2)
x_pca
#features2 = df[["income", "age", "loan"]]
#features2

model = Sequential()
model.add(Dense(10, input_dim=2, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))


