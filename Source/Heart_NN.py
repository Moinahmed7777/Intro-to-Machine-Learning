# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:49:38 2021

@author: Necro
"""

"""
dataset info
age: age in years
sex: 1 male, 0 female
cp: chest pain type
-- Value 0: typical angina
-- Value 1: atypical angina
-- Value 2: non-anginal pain
-- Value 3: asymptomatic

trestbps: resting blood pressure (in mm Hg on admission to the hospital)
chol: serum cholestoral in mg/dl 
(An ideal total cholesterol level is lower than 200 mg/dL. Anything between 
200 and 239 mg/dL is borderline, and anything above 240 mg/dL is high.)
fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
*restecg: resting electrocardiographic results(ECG)
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach: maximum heart rate achieved
exang: exercise induced angina (1 = yes; 0 = no)
oldpeak = ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
-- Value 0: upsloping
-- Value 1: flat
-- Value 2: downsloping
ca: na 
-number of major vessels (0-4) colored by flourosopy
thal: na
target: (1 yes heart disease,0 no h disease? )
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense



pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/heart.csv")

print(df.head(10))
 
#sns.heatmap(df.isnull(),cbar=False, cmap='viridis')
#sns.pairplot(df,hue='target',corner=True)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='viridis',annot=True)

print(df['cp'].nunique())

y = df['target']
#print(y.head())
df.drop(['target'],axis=1,inplace=True)
print(df.head(10))

scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca.fit(scaled_X)
x_pca = pca.transform(scaled_X)
print(scaled_X.shape)
print(x_pca.shape)

#print(type(y))
y= np.array(y).reshape(-1, 1)
#print(type(y))
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c=y, cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')

print(pca.components_)
#print(df.columns)

df_comp = pd.DataFrame(pca.components_, columns=df.columns)
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')


#NN 
features = pca.components_

# we have 2 classes(target) so the labels will have 3 values
# first class: (1.,0.) second class: (0.,1.)...
encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(x_pca, targets, test_size=0.2)
###############################
"""
Sequential model 
Dense layer : (20 neurons)*, 2 for pc1&pc2(x_pca.shape==(X,2)) ,
2nd dense=2, (for output layer since target 0 and 1)* 
"""
###############################

model = Sequential()
model.add(Dense(32, input_dim=2, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=1000, verbose=2)
results = model.evaluate(test_features, test_targets, use_multiprocessing=True)

print("Training is finished... The loss and accuracy values are:")
print(results)
"""
epoch=10000
Epoch 9997/10000
8/8 - 0s - loss: 0.3654 - accuracy: 0.8388
Epoch 9998/10000
8/8 - 0s - loss: 0.3653 - accuracy: 0.8388
Epoch 9999/10000
8/8 - 0s - loss: 0.3653 - accuracy: 0.8388
Epoch 10000/10000
8/8 - 0s - loss: 0.3663 - accuracy: 0.8388
2/2 [==============================] - 0s 2ms/step - loss: 0.3646 - accuracy: 0.8197
Training is finished... The loss and accuracy values are:
[0.36460983753204346, 0.8196721076965332

epoch=5000
Epoch 4997/5000
8/8 - 0s - loss: 0.3568 - accuracy: 0.8347
Epoch 4998/5000
8/8 - 0s - loss: 0.3569 - accuracy: 0.8306
Epoch 4999/5000
8/8 - 0s - loss: 0.3570 - accuracy: 0.8347
Epoch 5000/5000
8/8 - 0s - loss: 0.3566 - accuracy: 0.8306
2/2 [==============================] - 0s 2ms/step - loss: 0.4967 - accuracy: 0.7869
Training is finished... The loss and accuracy values are:
[0.4967449903488159, 0.7868852615356445]

epoch=1000
Epoch 997/1000
8/8 - 0s - loss: 0.3951 - accuracy: 0.8140
Epoch 998/1000
8/8 - 0s - loss: 0.3953 - accuracy: 0.8140
Epoch 999/1000
8/8 - 0s - loss: 0.3951 - accuracy: 0.8140
Epoch 1000/1000
8/8 - 0s - loss: 0.3952 - accuracy: 0.8140
2/2 [==============================] - 0s 997us/step - loss: 0.3795 - accuracy: 0.8033
Training is finished... The loss and accuracy values are:
[0.3794618844985962, 0.8032786846160889]

"""


