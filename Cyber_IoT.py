# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:50:12 2019

@author: Abhishek R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dataset=pd.read_csv('mainSimulationAccessTraces.csv')
x=Dataset.iloc[:,:-2].values
y=Dataset.iloc[:,12].values

#missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='constant',verbose=0)
imputer=imputer.fit(x[:,[8]])
x[:,[8]]=imputer.transform(x[:,[8]])

imputer1=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
imputer1=imputer1.fit(x[:,[10]])
x[:,[10]]=imputer1.transform(x[:,[10]])

z=pd.DataFrame(x[:,2])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(0,10):
    x[:,i] = labelencoder_X.fit_transform(x[:,i])
x=np.array(x,dtype=np.float)

y=labelencoder_X.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred) 



