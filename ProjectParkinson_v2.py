# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:26:38 2019

@author: Desktop-TL
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from readprocess_tappy import process_user

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import warnings; warnings.filterwarnings('ignore')

# %% Directories
directory_tappy = "Tappy Data/"
directory_user = "Archived users/"

# %% Data Import
user_data = os.listdir(directory_user)
tappy_data = os.listdir(directory_tappy)

ID_data = [x[5:15] for x in user_data]  # Get user ID from file name
ID_tappy = [x[0:10] for x in tappy_data] 

users_actual = [value for value in ID_data if value in ID_tappy]  # If the user has no Tappy data, we dont need that user

# Import User Data
columns = ['BirthYear', 'Gender', 'Parkinsons', 'Tremors', 'DiagnosisYear',
           'Sided', 'UPDRS', 'Impact', 'Levadopa', 'DA', 'MAOB', 'Other']

user_df = pd.DataFrame(columns=columns)
for userid in users_actual:
    f = open(directory_user+'User_'+userid+'.txt')
    user_df.loc[userid] = [line.split(': ')[1][: -1] for line in f.readlines()]
    f.close()

user_df['BirthYear'] = pd.to_numeric(user_df['BirthYear'], errors='coerce')
user_df['DiagnosisYear'] = pd.to_numeric(user_df['DiagnosisYear'], errors='coerce')
user_df = user_df.rename(index=str, columns={'Gender': 'Male'})  # renaming `Gender` to `Female`
user_df['Male'] = user_df['Male'] == 'Male'  # change string data to boolean data
user_df['Male'] = user_df['Male'].astype(int)  # change boolean data to binary data
str_to_bin_columns = ['Parkinsons', 'Tremors',
                      'Levadopa', 'DA', 'MAOB',
                      'Other'] # columns to be converted to binary data

for column in str_to_bin_columns:
    user_df[column] = user_df[column] == 'True'
    user_df[column] = user_df[column].astype(int)

user_df.loc[
    (user_df['Impact'] != 'Medium') &
    (user_df['Impact'] != 'Mild') &
    (user_df['Impact'] != 'Severe'), 'Impact'] = 'None'

to_dummy_column_indices = ['Sided', 'UPDRS']  # columns to be one-hot encoded
for column in to_dummy_column_indices:
    user_df = pd.concat([
        user_df.iloc[:, : user_df.columns.get_loc(column)],
        pd.get_dummies(user_df[column], prefix=str(column)),
        user_df.iloc[:, user_df.columns.get_loc(column) + 1:]
    ], axis=1)


# Import Tappy Data                    
column_names = [first_hand + second_hand + '_' + time for first_hand in ['L', 'R', 'S'] for second_hand in ['L', 'R', 'S'] for time in ['Hold time', 'Latency time', 'Flight time']]

user_tappy_df = pd.DataFrame(columns=column_names)

for user_id in user_df.index:
    user_tappy_data = process_user(directory_tappy, str(user_id), tappy_data)
    user_tappy_df.loc[user_id] = user_tappy_data

user_tappy_df = user_tappy_df.fillna(0)
user_tappy_df[user_tappy_df < 0] = 0
    
# %% Machine Learning
target = 'Parkinsons'
features = column_names
            
X = user_tappy_df[features]
X_scaled = StandardScaler().fit_transform(X)
y = user_df[target]

for col in X:
    X[X[col]==0] = X[col].mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test = train_test_split(X, y, test_size=0.3,
                                                                                random_state=42,
                                                                                stratify=y)

logreg = LogisticRegression()
logreg.fit(X_scaled_train, y_scaled_train)
y_pred_logreg = logreg.predict(X_scaled_test)

KNN = KNeighborsClassifier(n_neighbors=9)
KNN.fit(X_train, y_train)
y_pred_KNN = KNN.predict(X_test)

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

cvs_logreg = cross_val_score(logreg, X, y, scoring='accuracy', cv=5)
cvs_KNN = cross_val_score(KNN, X, y, scoring='accuracy', cv=5)
cvs_tree = cross_val_score(tree, X, y, scoring='accuracy', cv=5)

acc_logreg = accuracy_score(y_scaled_test, y_pred_logreg)
acc_KNN = accuracy_score(y_test, y_pred_KNN)
acc_tree = accuracy_score(y_test, y_pred_tree)

print(cvs_logreg, acc_logreg)
print(cvs_KNN, acc_KNN)
print(cvs_tree, acc_tree)

intercept = logreg.intercept_
coefs = logreg.coef_
print(coefs)
# %% Other
probs_logreg = logreg.predict_proba(X_scaled_test)
probs_KNN = KNN.predict_proba(X_test)
probs_tree = tree.predict_proba(X_test)

fpr_logreg, tpr_logreg, threshold_logreg = roc_curve(y_scaled_test, probs_logreg[:,1])
fpr_KNN, tpr_KNN, threshold_KNN = roc_curve(y_test, probs_KNN[:,1])
fpr_tree, tpr_tree, threshold_tree = roc_curve(y_test, probs_tree[:,1])

plt.plot(fpr_logreg, tpr_logreg, color='red', label='logreg')
plt.plot(fpr_KNN, tpr_KNN, color='green', label='KNN')
plt.plot(fpr_tree, tpr_tree, color='blue', label='tree')
plt.plot([0,1], [0,1], color='black', linestyle='--', label='no skill ref')
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.title('Classifier Comparison')
plt.legend()
