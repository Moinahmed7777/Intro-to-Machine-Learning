# -*- coding: utf-8 -*-
"""
Created on Sun May  9 23:56:01 2021

@author: Necro
"""
'''Libraries'''
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense

pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/titanic.csv")
'''EDA'''
print(df.head(20))
print(df.info())

'''from df.info() we can see the missing data by Non-Null count 
and therefore dropping cabin,name,boat,body,home.dest,ticket '''
df.drop(['cabin','name','boat','body','home.dest','ticket'], axis=1, inplace=True)

'''Separating age column to fill it up with Pclass later'''
age_col = df["age"]

'''fill age with Pclass built in pandas apply>for loops (RTime)'''
def fill_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
    
df['age']=df[['age','pclass']].apply(fill_age, axis=1)

'''final check before dropping Null rows'''   
print(df.info())

df.dropna(inplace=True)

'''Get encoding for the sex and embarked columns into separate numerical cols '''
X = pd.get_dummies(data=df)

'''setting aside the target column as y and dropping from X'''
y= X['survived']
X.drop(['survived'],axis=1,inplace=True)

'''scaling the data with standard scaler and creating the 2 Principal components
and fitting,transforming the scaled data '''
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
pca = PCA(n_components=2)
pca.fit(scaled_X)
x_pca = pca.transform(scaled_X)

'''checking shapes of the scaled_X and x_pca'''
print(scaled_X.shape)
print(x_pca.shape)

'''Converting y to numpy array'''
y = np.array(y).reshape(-1, 1)

'''Plotting the 2 principal components with c as the target/y'''
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c=y, cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')

'''Heatmap for visualizing the correlation of the Pcs with the rest of the data'''
df_comp = pd.DataFrame(pca.components_, columns=X.columns)
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')

'''Nueral Network intro''' 
features = pca.components_

''' we have 3 classes(target) so the labels will have 3 values
first class: (1.,0.,0.) second class: (0.,1.,0.)...'''
encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()


train_features, test_features, train_targets, test_targets = train_test_split(x_pca, targets, test_size=0.2)

"""Sequential model
single layer stacking 

2 Dense layer
1st Dense layer(input): 20 neurons, 2 for pc1&pc2 ,with activation function 'sigmoid'
2nd Dense layer(output): 2 for output values, with activation function 'softmax'

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
model.add(Dense(20, input_dim=2, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=1000, verbose=2)

"""epoch=10000
Epoch 9999/10000
33/33 - 0s - loss: 0.4309 - accuracy: 0.8228
Epoch 10000/10000
33/33 - 0s - loss: 0.4312 - accuracy: 0.8238
9/9 [==============================] - 0s 499us/step - loss: 0.5702 - accuracy: 0.7405
Training is finished... The loss and accuracy values are:
[0.570188581943512, 0.7404580116271973]

##epoch=100
Epoch 97/100
33/33 - 0s - loss: 0.4865 - accuracy: 0.7854
Epoch 98/100
33/33 - 0s - loss: 0.4860 - accuracy: 0.7816
Epoch 99/100
33/33 - 0s - loss: 0.4862 - accuracy: 0.7835
Epoch 100/100
33/33 - 0s - loss: 0.4862 - accuracy: 0.7883
9/9 [==============================] - 0s 499us/step - loss: 0.4803 - accuracy: 0.8015
Training is finished... The loss and accuracy values are:
[0.4803255498409271, 0.8015267252922058]

#epoch = 1000
Epoch 997/1000
33/33 - 0s - loss: 0.4796 - accuracy: 0.7826
Epoch 998/1000
33/33 - 0s - loss: 0.4792 - accuracy: 0.7845
Epoch 999/1000
33/33 - 0s - loss: 0.4793 - accuracy: 0.7893
Epoch 1000/1000
33/33 - 0s - loss: 0.4799 - accuracy: 0.7835
9/9 [==============================] - 0s 499us/step - loss: 0.4576 - accuracy: 0.8168
Training is finished... The loss and accuracy values are:
[0.4576311409473419, 0.8167939186096191]
"""

results = model.evaluate(test_features, test_targets, use_multiprocessing=True)

print("Training is finished... The loss and accuracy values are:")
print(results)







