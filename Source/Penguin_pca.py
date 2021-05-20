# -*- coding: utf-8 -*-
"""
Created on Sat May  8 21:18:24 2021

@author: Necro
"""
'''Libraries'''

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense

'''display all columns, no restraint, and df creation'''
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/penguins_size.csv")


'''Exploratory Data Analysis'''

print(df.head())

'''check for null,and data types'''

print(df.info())

'''drop all rows with null values'''

df.dropna(inplace=True)

'''df2 to separate the species col for later use'''

df2= df["species"]

'''drop species from df'''

df.drop(["species"],axis=1,inplace = True)

'''Trgt_n converts species column stored in df2 into np.array'''

Trgt_n =np.array(df2.values.tolist())


'''purpose of X, encodes for the features of islands into 
3 separate columns, converts object to numerical cols'''

X = pd.get_dummies(data=df)

''' drop invalid data's in columns'''
X.drop(["sex_."],axis=1,inplace=True)

'''to check if everythings alright'''
print(X.info())


"""PCA analysis"""            
'''scale the data,scale X, fit and transform'''
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

'''creating 2 Principal Components, fitting and transform the scaled X data'''  
pca = PCA(n_components=2)
pca.fit(scaled_X)
x_pca = pca.transform(scaled_X)

'''check for the shapes, x_pca will have (X,2) since 2 components are used'''
print(scaled_X.shape)
print(x_pca.shape)


''' A for loop to encode the species column into numerical values
Adelie=1
Chinstrap=2
Gentoo=3
n_l will contain all the rows of converted species rows as a list
y will contain the convert list it to numpy array.
'''
n_l =[]
for i in range(len(Trgt_n)):
    if Trgt_n[i]=="Adelie":
        n_l.append(1)
    elif Trgt_n[i]=='Chinstrap':
        n_l.append(2)
    elif Trgt_n[i]=='Gentoo':
        n_l.append(3)

y = np.array(n_l).reshape(-1,1)

''' plot for the principal components with its target as the data_arr'''
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c=y, cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')

''' Heatmap to see how correlated features are with the principal components'''

df_comp = pd.DataFrame(pca.components_, columns=X.columns)
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')

"""Nueral Network introduction"""
features = pca.components_

'''we have 3 classes(target) so the labels will have 3 values
first class: (1.,0.,0.) second class: (0.,1.,0.)...'''

encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

'''test,train features for NN'''
train_features, test_features, train_targets, test_targets = train_test_split(x_pca, targets, test_size=0.2)

"""Sequential model
single layer stacking 

2 Dense layer
1st Dense layer(input): 32 neurons, 2 for pc1&pc2 ,with activation function 'sigmoid'
2nd Dense layer(output): 3 for output values, with activation function 'softmax'

activation:- 
sigmoid: to encode the sum of weights of all nuerons to either 0 or 1.
softmax: activation function used in the output layer which will take 
the highest probable unit with the largest input has output +1
while all other units have output 0.
    
optimizer:
Adam,combines Adagrad & RMSprop, combination of adaptive gradient optimizer(Adagrad) which uses 
1st order derivatives and 2nd order derivative (RMSprop) which uses squared derivatives.
Adagrad, adjusts step size 
RMSprop, makes step size smaller when closer to global minimum.  
    
*categorical_crossentropy: Used as a loss function for multi-class classification model 
where there are two or more output labels.sparse_categorical_crossentropy, same as categorical
but different interface.
binary_crossentropy: Used as a loss function for binary classification model.
metrics=['accuracy']: infered from the loss function used. Should not be valid on 
regression models(R2,RMSE..should be used?) only classification task.
    
use_multiprocessing: if True then use process-based threading"""
    
model = Sequential()
model.add(Dense(32, input_dim=2, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=1000, verbose=2)

results = model.evaluate(test_features, test_targets, use_multiprocessing=True)

print("Training is finished... The loss and accuracy values are:")

print(results)
