# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:45:03 2018

@author: ThapaRak
"""

# Data ananlysis and wrangling
import pandas as pd
import numpy as np

train = r'C:\Users\ThapaRak\Documents\MyCodes\ML_Exercises\dataset\Titanic_Kaggle\data\train.csv'
test = r'C:\Users\ThapaRak\Documents\MyCodes\ML_Exercises\dataset\Titanic_Kaggle\data\test.csv'

train_df = pd.read_csv(train)
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

most_frequent = train_df.Embarked.mode()
most_frequent = most_frequent[0]
train_df['Embarked'].fillna(most_frequent, inplace=True)

# Converting String to character.
train_df['Embarked'] = train_df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

# Map Gender column
train_df['Sex']=train_df['Sex'].map({'male':0, 'female':1}).astype(int)

# Removing the missing values for all coumns.
def num_missing(x):
    return sum(x.isnull())

# Implementation 1: Remove entire columns when values go missing
train = train_df.drop(columns=['Age', 'Cabin'])

# Drop column name
train = train.drop(columns=['Name'])

# Drop column Ticket
train = train.drop(columns=['Ticket'])

# Map Gender column
#train['Sex']=train['Sex'].map({'male':0, 'female':1}).astype(int)

# drop the passengerid column
train=train.drop(columns=['PassengerId'])

# Modelling 1:
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]

# Random Forest Classifier
# Using Decision Tree and Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
accuracy = round(random_forest.score(X_train, Y_train)*100, 2)
print('Accuracy1:', accuracy)
# Random forest with estimators
random_forest = RandomForestClassifier(n_estimators=2000, max_depth=10000)
random_forest.fit(X_train, Y_train)
accuracy = round(random_forest.score(X_train, Y_train)*100, 2)
print('Accuracy1:', accuracy)