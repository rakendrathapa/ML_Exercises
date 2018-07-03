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

# Adding One Hot Encoding
whole_1_hot = pd.get_dummies(X_train, columns=["Pclass", "Embarked", "Title"])

# Splitting the Data.
#msk = np.random.rand(len(X_train)) < 0.8
msk = np.random.rand(len(whole_1_hot)) <= 0.8

X_train_1 = whole_1_hot[msk]
X_test_1 = whole_1_hot[~msk]

Y_train_1 = Y_train[msk]
Y_test_1 = Y_train[~msk]


# Normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_norm = min_max_scaler.fit_transform(X_train_1)
X_test_norm = min_max_scaler.transform(X_test_1)

X_train_norm_df = pd.DataFrame(X_train_norm, index=X_train_1.index, columns=X_train_1.columns)
X_test_norm_df = pd.DataFrame(X_test_norm, index=X_test_1.index, columns=X_test_1.columns)

# machine learning
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=10, max_depth=10,  max_features = .7, verbose = 0, max_leaf_nodes = 10, random_state=0 )
random_forest.fit(X_train_norm, Y_train_1)

train_accuracy = round(random_forest.score(X_train_norm, Y_train_1)*100,2)
test_accuracy = round(random_forest.score(X_test_norm, Y_test_1)*100, 2)

print('1. Random Forest: Train Accuracy:%f \t Test Accuracy:%f\n'% (train_accuracy, test_accuracy))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_norm, Y_train_1)

train_accuracy = round(logreg.score(X_train_norm, Y_train_1)*100,2)
test_accuracy = round(logreg.score(X_test_norm, Y_test_1)*100, 2)

print('2. Logistic Regression: Train Accuracy:%f \t Test Accuracy:%f\n'% (train_accuracy, test_accuracy))

# Support Vector Machines
from sklearn.svm import SVC
svc =  SVC()
svc.fit(X_train_norm, Y_train_1)
train_accuracy = round(svc.score(X_train_norm, Y_train_1)*100,2)
test_accuracy = round(svc.score(X_test_norm, Y_test_1)*100, 2)

print('3. Support Vector Machines: Train Accuracy:%f \t Test Accuracy:%f\n'% (train_accuracy, test_accuracy))

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train_norm, Y_train_1)
train_accuracy = round(gaussian.score(X_train_norm, Y_train_1)*100,2)
test_accuracy = round(gaussian.score(X_test_norm, Y_test_1)*100, 2)

print('4. Gaussian Naive Bayes: Train Accuracy:%f \t Test Accuracy:%f\n'% (train_accuracy, test_accuracy))
