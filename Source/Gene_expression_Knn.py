# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:33:39 2021

@author: Necro
"""


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv("../Datasets/gene_expression.csv")

#scatter plot 
sns.scatterplot(x ='Gene One', y ='Gene Two', hue ='Cancer Present', data=df, alpha=0.7)

#for seeing the overlapping 
sns.scatterplot(x ='Gene One', y ='Gene Two', hue ='Cancer Present', data=df)
plt.xlim(2,6)
plt.ylim(3,10)
plt.legend(loc=(1.1,0.5))

#usual train test split
X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state = 42)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
print("#############################################")
print(X_test.head(30))
print("#############################################")
print(scaled_X_test)
print("#############################################")

#print("info scaled", scaled_X_test.describe())
acc=0
for i in range( 0,len(scaled_X_test)):
    acc = scaled_X_test[i][1] + acc

len1 = len(scaled_X_test)
print(len1)
print("avg", acc/len1)
#KN
knn_model = KNeighborsClassifier(n_neighbors=22)
knn_model.fit(scaled_X_train,y_train)
y_pred = knn_model.predict(scaled_X_test)

''' if precision and recall are close to accuracy then the data set is balanced.
Since we can tweak K in 'n_neighbors=1' to increase our model's accuracy we will do a for loop.
'''
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
#print(classification_report(y_test, y_pred))

# for the elbow graph to visually see which K would be best.
test_error_rates = []

for k in range(1,40):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_X_train,y_train)
    y_pred = knn_model.predict(scaled_X_test)
    
    test_error = 1 - accuracy_score(y_test, y_pred)
    test_error_rates.append(test_error)
    
plt.figure(figsize=(10,6),dpi=200)
plt.plot(range(1,40), test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel('K Value')


#pipeline for overfitting underfitting, will find the K for us, another auto technique
scaler = StandardScaler()
knn = KNeighborsClassifier()
operations = [('scaler',scaler),('knn',knn)]
pipe = Pipeline(operations)
k_values = list(range(1,20))
param_grid = {'knn__n_neighbors': k_values}
full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
full_cv_classifier.fit( X_train, y_train)

#print(full_cv_classifier.best_estimator_.get_params())

#random test
new_patient = [[3.8,6.4]]
#print(full_cv_classifier.predict(new_patient))
