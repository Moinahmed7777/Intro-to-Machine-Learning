# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:08:28 2021

@author: Necro
"""

import time
import numpy as np
import pandas as pd 
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error




pd.set_option('display.expand_frame_repr', False) 
#pd.set_option("display.max_rows", None, "display.max_columns", None)


dt = datetime.today()
#Read in bikeshare.csv file and set it to a dataframe called bike
bike = pd.read_csv("../Datasets/bikeshare.csv")
bike["datetime"] = pd.to_datetime(bike["datetime"])
bike["Hour"] = bike["datetime"].apply(lambda time: time.hour)


#sns.pairplot(bike,corner=True)

#Check the head of df
print('the head of df :')
print(bike.head())

#3 weather? count/rentals?


#plots
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(24,8))
sns.scatterplot(x=bike["temp"],y=bike["count"],alpha=(0.75),data=bike, ax=ax1)
pd.set_option('display.expand_frame_repr', False)
sns.scatterplot(x=bike["datetime"],y=bike["count"],hue=bike["temp"],data=bike, ax=ax2)
sns.boxplot(y=bike["count"],x=bike["season"])

#correlation between temp and count
correlation = bike.corr()
print('correlation between temp and count :')
print(correlation['count']['temp'])


X1 = bike["temp"]
#2D to 1D
X = np.asarray(X1).reshape((-1, 1))
y = bike["count"]


X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.4,random_state=101)

temp_model = LinearRegression()

scaler = StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test =scaler.transform(X_test)


temp_model.fit(scaled_X_train,y_train)


y_pred = temp_model.predict(scaled_X_test)
#prediction of count at temp 25 for tempvscount linear Regression
campaign = [[25]]
print("Prediction of count at temp 25 : ",temp_model.predict(campaign))

#summary
print('ypred',y_pred)
print("linear regression model coeficient",temp_model.coef_)

#Quadratic has better scores
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
MAE =mean_absolute_error(y_test,y_pred)

MSE = mean_squared_error(y_test,y_pred)
print("temp_count linear Regression RMSE",rmse)
print("temp_count linear Regression R^2",r2)
print("MAE :", MAE)
print("MSE :", MSE)


X_13=bike.drop(["datetime","atemp","casual","registered","count"],axis=1)
y_13 = bike["count"]

#X_train, X_test, y_train,y_test= train_test_split(X_13,y_13,test_size=0.4,random_state=101)


temp_model_13 = LinearRegression()
scaler = StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test =scaler.transform(X_test)


temp_model_13.fit(scaled_X_train,y_train)

y_pred_13 = temp_model_13.predict(scaled_X_test)

#summary
MAE =mean_absolute_error(y_test,y_pred_13)
MSE = mean_squared_error(y_test,y_pred_13)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_13))
#r2 = r2_score(y_test,y_pred_13)

print("MAE for new:", MAE)
print("MSE for new:", MSE)
print("temp_model RMSE",rmse)
#print("all_count linear Regression R^2",r2)


#will try to increase the degree to see how the RMSE changes

#Train_err= []
#Test_err= []
for n in range(2,6):
    polynomial_features= PolynomialFeatures(degree=n)
    #x_poly = polynomial_features.fit_transform(X_train)
    x_poly = polynomial_features.fit_transform(X_13)
    
    X_train2, X_test2, y_train2,y_test2= train_test_split(x_poly,y_13,test_size=0.4,random_state=101)
  
    model = LinearRegression() #fit
    
    model.fit(X_train2, y_train2)
    
    X_test_pred  = model.predict(X_test2)
    X_train_pred = model.predict(X_train2)
    
    #Train
    MAE = mean_absolute_error(y_train2,X_train_pred)
    MSE = mean_squared_error(y_train2,X_train_pred)
    RMSE_train =np.sqrt(MSE)
    
    #Test
    MAE = mean_absolute_error(y_test2, X_test_pred)
    MSE = mean_squared_error(y_test2,  X_test_pred)
    RMSE_test =np.sqrt(MSE)
    
    
    #Train_err.append(RMSE_train)
    #Test_err.append(RMSE_test)
    #rmse = np.sqrt(mean_squared_error(y_test2,y_poly_pred))
    #r2 = r2_score(y_train,y_poly_pred)
    print("polynomial regression RMSE with degree ",n," on training set : ", RMSE_train, " testing set :", RMSE_test)

#plt.plot(Train_err[:5],label="Train")
#plt.plot(Test_err[:5],label="Test")
#plt.xlabel("flex model") 
#plt.ylabel("RMSE")
#plt.legend
#plt.show()
    

# By looking at the polynomial regression at degree 3 , 4 , 5. Degree 4 will be suitable choice for the degree although the error difference are not that much, since it performs better , we can see the degree 5 performs worst on testing set data.
# The Linear model had bad accuracy and RMSE score but the r2 score tells us this model will perform better in polynomial Regression,
# therefore after testing with the training set the score(RMSE,r2) of the polynomial regression gets better with higher degrees as we can see in the for loop. 