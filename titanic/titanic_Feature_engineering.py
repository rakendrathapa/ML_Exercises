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

# . Add null Ages
guess_ages = np.zeros((2,3))
for i in range(2):      # Gender
    for j in range(3):  # PClass
        guess_df = train_df[(train_df['Sex'] == i) & \
                          (train_df['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess_df.median()
        
        # Convert random age float to nearing .5 age
        guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5

for i in range(2):
    for j in range(3):
        train_df.loc[(train_df.Age.isnull()) & (train_df.Sex == i) & (train_df.Pclass==j+1),\
                     'Age'] = guess_ages[i, j]
          

# Feature Engineering
# 1. Get the Title. Editing the Title
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['Title'] = train_df['Title'].replace(['Capt', 'Col','Countess', 'Don','Dr', 'Major', 'Rev', 'Sir'], 'Hon')

train_df['Title'] = train_df['Title'].replace('Jonkheer', 'Mr')
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Lady', 'Mrs')

train_df['Title'] = train_df['Title'].map({"Miss": 1, "Master": 2, "Mrs": 3, "Mr": 4, "Hon": 5})

X_train =  train_df.drop(columns=['Cabin', 'Name', 'Ticket','Survived'])

Y_train = train_df["Survived"]

X_train  = X_train.drop("PassengerId", axis=1)

msk = np.random.rand(len(X_train)) < 0.8

X_train_1 = X_train[msk]
X_test_1 = X_train[~msk]

Y_train_1 = Y_train[msk]
Y_test_1 = Y_train[~msk]

# machine learning
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=10, max_depth=10,  max_features = .7, verbose = 0, max_leaf_nodes = 10, random_state=0 )
random_forest.fit(X_train_1, Y_train_1)

train_accuracy = round(random_forest.score(X_train_1, Y_train_1)*100,2)
test_accuracy = round(random_forest.score(X_test_1, Y_test_1)*100, 2)

print('Train Accuracy:%f \t Test Accuracy:%f\n'% (train_accuracy, test_accuracy))

# Model Interpretation
importances = random_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0); 
indices = np.argsort(importances)[::-1]

# print feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature%d %s (%f)" % (f + 1, indices[f], X_train.columns[indices[f]], importances[indices[f]]))

import matplotlib.pyplot as plt
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()