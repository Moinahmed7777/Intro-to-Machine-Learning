# -*- coding: utf-8 -*-
"""
Created on Tue May 11 02:45:03 2021

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
"""Libraries"""
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report,plot_confusion_matrix 
from sklearn.preprocessing import MinMaxScaler


pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv("../Datasets/heart.csv")
'''EDA'''
#print(df.head(10))

'''Visualize Null values and pairplot of all features to features, and 
Heatmap for visualizing the correlation''' 
sns.heatmap(df.isnull(),cbar=False, cmap='viridis')
sns.pairplot(df,hue='target',corner=True)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='viridis',annot=True)

'''Setting target column to y, and droping from df'''
y = df['target']
df.drop(['target'],axis=1,inplace=True)

'''Scaling data with Standard scaler'''
scaler = StandardScaler()
scaled_X = scaler.fit_transform(df)

'''Creating 2 Principal Components, fit and transforming'''
pca = PCA(n_components=2)
pca.fit(scaled_X)
x_pca = pca.transform(scaled_X)

'''Checking shapes of scaled_X and x_pca'''
print(scaled_X.shape)
print(x_pca.shape)

'''converting y series to numpy array'''
y= np.array(y)

'''Ploting the PCs and Heatmap for the correlation of the features to PCs'''
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c=y, cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')

df_comp = pd.DataFrame(pca.components_, columns=df.columns)
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')

#encoder = OneHotEncoder()
#targets = encoder.fit_transform(y).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(x_pca, y, test_size=0.2)

'''report funtion print the classification report of the passed model
and plots the confusion matrixes'''
def report(model):
    preds = model.predict(test_features)
    print(classification_report(test_targets,preds))
    plot_confusion_matrix(model,test_features,test_targets)

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

scaler2 = MinMaxScaler()
scaled_X_MM = scaler2.fit_transform(train_features)

mnb.fit(scaled_X_MM,train_targets)

print("MultinomialNB MODEL")
report(mnb)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(train_features,train_targets)

print("GuassianNB MODEL")
report(nb)


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(train_features,train_targets)

print("LogisticRegression")
report(log_model)


# LogisticRegressionCV

logCV = LogisticRegressionCV()
logCV.fit(train_features,train_targets);

print("LogisticRegressionCV")
report(logCV)

# LinearSVC
from sklearn.svm import LinearSVC
linearsvc = LinearSVC()
linearsvc.fit(train_features,train_targets)

print("LinearSVC")
report(linearsvc)



from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(train_features,train_targets)
print("DECISION TREE MODEL")
report(decisiontree)


# SVM
from sklearn.svm import SVC
svc = SVC()
svc.fit(train_features,train_targets)

print("Support Vector Machine")
report(svc)

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
test_error_rates = []

for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_features,train_targets) 
   
    y_pred_test = knn_model.predict(test_features)
    
    test_error = 1 - accuracy_score(test_targets,y_pred_test)
    test_error_rates.append(test_error)
    
plt.figure(figsize=(10,6),dpi=200)
plt.plot(range(1,30),test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel("K Value")

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=6)
knn_model.fit(train_features,train_targets)

print("KNN MODEL")
report(knn_model)

