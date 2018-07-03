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
test_df = pd.read_csv(test)

most_frequent_train = train_df.Embarked.mode()
most_frequent_train = most_frequent_train[0]
train_df['Embarked'].fillna(most_frequent_train, inplace=True)

most_frequent_test = train_df.Embarked.mode()
most_frequent_test = most_frequent_test[0]
test_df['Embarked'].fillna(most_frequent_test, inplace=True)

# Data Preprocessing
# Converting String to character.
train_df['Embarked'] = train_df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
test_df['Embarked'] = test_df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

# Map Gender column
train_df['Sex']=train_df['Sex'].map({'male':0, 'female':1}).astype(int)
test_df['Sex']=test_df['Sex'].map({'male':0, 'female':1}).astype(int)

# Train Data - Add for null Ages
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

# Test Data - Add for null Ages
guess_ages = np.zeros((2,3))
for i in range(2):      # Gender
    for j in range(3):  # PClass
        guess_df = test_df[(test_df['Sex'] == i) & \
                          (test_df['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess_df.median()
        
        # Convert random age float to nearing .5 age
        guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5

for i in range(2):
    for j in range(3):
        test_df.loc[(test_df.Age.isnull()) & (test_df.Sex == i) & (test_df.Pclass==j+1),\
                     'Age'] = guess_ages[i, j]					 

# Add for the missing Fare Value
guess_fare = np.zeros((3,3))
for i in range(3):      # Embarked
    for j in range(3):  # PClass
        guess_df = test_df[(test_df['Embarked'] == i) & \
                          (test_df['Pclass'] == j+1)]['Fare'].dropna()
        fare_guess = guess_df.median()
        
        # Convert random age float to nearing .5 age
        guess_fare[i, j] = int(fare_guess/0.5 + 0.5) * 0.5

for i in range(3):
    for j in range(3):
        test_df.loc[(test_df.Fare.isnull()) & (test_df.Embarked == i) & (test_df.Pclass==j+1),\
                     'Fare'] = guess_fare[i, j]
        
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

# 1. Title Editing for Test Data
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Title'].replace(['Capt', 'Col','Countess', 'Don','Dr', 'Major', 'Rev', 'Sir'], 'Hon')
test_df['Title'] = test_df['Title'].replace('Jonkheer', 'Mr')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Lady', 'Mrs')
test_df['Title'] = test_df['Title'].map({"Miss": 1, "Master": 2, "Mrs": 3, "Mr": 4, "Hon": 5})

X_train =  train_df.drop(columns=['Cabin', 'Name', 'Ticket','Survived'])
X_test =  test_df.drop(columns=['Cabin', 'Name', 'Ticket'])

Y_train = train_df["Survived"]

X_train  = X_train.drop("PassengerId", axis=1)
X_test = X_test.drop("PassengerId", axis=1)

# Adding One Hot Encoding
X_train_hot = pd.get_dummies(X_train, columns=["Pclass", "Embarked", "Title"])
X_test_hot  = pd.get_dummies(X_test , columns=["Pclass", "Embarked", "Title"])

# Normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

X_train_norm = min_max_scaler.fit_transform(X_train_hot)
X_test_norm  = min_max_scaler.fit_transform(X_test_hot)

# machine learning
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=10, max_depth=10,  max_features = .7, verbose = 0, max_leaf_nodes = 10, random_state=0 )
random_forest.fit(X_train_norm, Y_train)

train_accuracy = round(random_forest.score(X_train_norm, Y_train)*100,2)
test_predict = random_forest.predict(X_test_norm)
print('1. Random Forest: Train Accuracy:%f' % (train_accuracy))

data = {'PassengerId':test_df.PassengerId, 'Survived': test_predict}
df = pd.DataFrame(data)
df.to_csv('rf_gender_submission.csv', index=False)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_norm, Y_train)

train_accuracy = round(logreg.score(X_train_norm, Y_train)*100,2)
test_predict = logreg.predict(X_test_norm)

print('2. Logistic Regression: Train Accuracy:%f' % (train_accuracy))

data = {'PassengerId':test_df.PassengerId, 'Survived': test_predict}
df = pd.DataFrame(data)
df.to_csv('lg_gender_submission.csv', index=False)

# Support Vector Machines
from sklearn.svm import SVC
svc =  SVC()
svc.fit(X_train_norm, Y_train)

train_accuracy = round(svc.score(X_train_norm, Y_train)*100,2)
test_predict = svc.predict(X_test_norm)

print('3.State Vector Machine: Train Accuracy:%f' % (train_accuracy))

data = {'PassengerId':test_df.PassengerId, 'Survived': test_predict}
df = pd.DataFrame(data)
df.to_csv('svc_gender_submission.csv', index=False)