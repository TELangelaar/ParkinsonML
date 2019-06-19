# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:11:34 2019

@author: s2589656
"""
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from readprocess_tappy import process_user

import warnings; warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

# %% Methods
def model_pipeline(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test


def model_evaluation(model, X, y, y_pred):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    cvs = cross_val_score(model, X, y, scoring='accuracy', cv=5)
    acc = accuracy_score(y_test, y_pred)
    probs = model.predict_proba(X_test)
    fpr, tpr, threshold = roc_curve(y_test, probs[:, 1])
    auc = roc_auc_score(y_test, probs[:, 1])
    conf = confusion_matrix(y_test, y_pred)
    return cvs, acc, auc, fpr, tpr, conf, threshold, probs

# %% Directories
directory_tappy = "Tappy Data/"
directory_user = "User Data/"

# %% Data Import
user_data = os.listdir(directory_user)
tappy_data = os.listdir(directory_tappy)

ID_data = [x[5:15] for x in user_data] #Get user ID from file name
ID_tappy = [x[0:10] for x in tappy_data]   

users_actual = [value for value in ID_data if value in ID_tappy] #If the user has no Tappy data, we dont need that user

# Import User Data
medicines = ['Levadopa', 'DA', 'MAOB', 'Other']
columns = ['BirthYear', 'Gender', 'Parkinsons', 'Tremors', 'DiagnosisYear',
    'Sided', 'UPDRS', 'Impact', 'Levadopa', 'DA', 'MAOB', 'Other']
user_df = pd.DataFrame(columns=columns)
for userid in users_actual:
    f = open(directory_user+'User_'+userid+'.txt')
    user_df.loc[userid] = [line.split(': ')[1][: -1] for line in f.readlines()]
    f.close()

# Clean User Data
user_df['BirthYear'] = pd.to_numeric(user_df['BirthYear'], errors='coerce')
user_df['DiagnosisYear'] = pd.to_numeric(user_df['DiagnosisYear'], errors='coerce')
user_df = user_df.rename(index=str, columns={'Gender': 'Male'}) # renaming `Gender` to `Female`
user_df['Male'] = user_df['Male'] == 'Male' # change string data to boolean data
user_df['Male'] = user_df['Male'].astype(int) # change boolean data to binary data
str_to_bin_columns = ['Parkinsons', 'Tremors', 'Levadopa', 'DA', 'MAOB', 'Other'] # columns to be converted to binary data

for column in str_to_bin_columns:
    user_df[column] = user_df[column] == 'True'
    user_df[column] = user_df[column].astype(int)

user_df.loc[
    (user_df['Impact'] != 'Medium') &
    (user_df['Impact'] != 'Mild') &
    (user_df['Impact'] != 'Severe'), 'Impact'] = 'No'

to_dummy_column_indices = ['Sided', 'UPDRS','Impact'] # columns to be one-hot encoded
for column in to_dummy_column_indices:
    user_df = pd.concat([
        user_df.iloc[:, : user_df.columns.get_loc(column)],
        pd.get_dummies(user_df[column], prefix=str(column)),
        user_df.iloc[:, user_df.columns.get_loc(column) + 1 :]
    ], axis=1)

# NA data in DiagnosisYear and BirthYear
user_df['DiagnosisYear'] = user_df['DiagnosisYear'].fillna(0)
user_df['BirthYear'] = user_df['BirthYear'].fillna(0)

# Making everything an integer
for column in user_df:
    user_df[column] = user_df[column].astype(int)

# %% Import Tappy Data
tappy_names = ['L_Hand_mean', 'L_Hand_std', 'L_Hand_kurt', 'L_Hand_skew',
               'R_Hand_mean', 'R_Hand_std', 'R_Hand_kurt', 'R_Hand_skew', 
               'diff_Hand_mean',
               'LR_mean', 'LR_std', 'LR_kurt', 'LR_skew',
               'RL_mean', 'RL_std', 'RL_kurt', 'RL_skew',
               'LL_mean', 'LL_std', 'LL_kurt', 'LL_skew',
               'RR_mean', 'RR_std', 'RR_kurt', 'RR_skew',
               'diff_opposite_mean', 'diff_same_mean']

user_tappy_df = pd.DataFrame(columns=tappy_names)

for user_id in user_df.index:
    user_tappy_data = process_user(directory_tappy, str(user_id), tappy_data)
    user_tappy_df.loc[user_id] = user_tappy_data

# %% NaN values, outliers and merging DataFrames

# Mean imputation & removing outliers
for column in user_tappy_df:
    user_tappy_df[column].fillna(user_tappy_df[column].mean(), inplace=True)

full_set = pd.merge(user_tappy_df.reset_index(), user_df.reset_index(), on='index')
full_set.set_index('index', inplace=True)

full_set.to_csv('FullDataset.csv')
# %% Data preparation & EDA
features_hold = ['L_Hand_mean', 'L_Hand_std', 'L_Hand_kurt', 'L_Hand_skew',
                 'R_Hand_mean', 'R_Hand_std', 'R_Hand_kurt', 'R_Hand_skew',
                 'diff_Hand_mean']

features_lat = ['LR_mean', 'LR_std', 'LR_kurt', 'LR_skew',
                    'RL_mean', 'RL_std', 'RL_kurt', 'RL_skew',
                    'LL_mean', 'LL_std', 'LL_kurt', 'LL_skew',
                    'RR_mean', 'RR_std', 'RR_kurt', 'RR_skew',
                    'diff_opposite_mean', 'diff_same_mean']
target = 'Parkinsons'

# Select target group: Mild Parkinsons with no LDopa usage.
Y_park = full_set[full_set['Parkinsons'] == 1]
N_park = full_set[full_set['Parkinsons'] == 0]

Y_park_mild = Y_park[full_set['Impact_Mild'] == 1]
N_park_no = N_park[full_set['Impact_No'] == 1]

Y_park_mild_ldopa = Y_park_mild[full_set['Levadopa'] == 0]

pca_set = pd.concat([N_park_no, Y_park_mild_ldopa])

X_hold = pca_set[features_hold].values
X_lat = pca_set[features_lat].values

# Standardizing Data
X_hold = RobustScaler().fit_transform(X_hold)
X_lat = RobustScaler().fit_transform(X_lat)

# Target variable
y = pca_set['Parkinsons'].values

sns.pairplot(pca_set[features_hold])
plt.show()

sns.pairplot(pca_set[features_lat])
plt.show()

# %% PCA
pcahold_fit = PCA(n_components=0.9).fit(X_hold)
pcalat_fit = PCA(n_components=0.9).fit(X_lat)

pca_hold = pcahold_fit.transform(X_hold)
pca_lat = pcalat_fit.transform(X_lat)

columns_hold = ['PC'+str(x)+'_h' for x in range(len(pca_hold[0]))]
columns_lat = ['PC'+str(x)+'_l' for x in range(len(pca_lat[0]))]

pc_hold_df = pd.DataFrame(pca_hold, columns=columns_hold)
pc_lat_df = pd.DataFrame(pca_lat, columns=columns_lat)

plt.figure(figsize=(5, 5))
plt.plot(pcahold_fit.explained_variance_, 'o-')
plt.xlabel('number of components')
plt.ylabel('explained variance ratio')
plt.title('PCA on Hold features')
plt.show()
print(pcahold_fit.explained_variance_ratio_.cumsum())

plt.figure(figsize=(5, 5))
plt.plot(pcalat_fit.explained_variance_, 'o-')
plt.xlabel('number of components')
plt.ylabel('explained variance ratio')
plt.title('PCA on Lat features')
plt.show()
print(pcalat_fit.explained_variance_ratio_.cumsum())

sns.pairplot(pc_hold_df)
plt.show()

sns.pairplot(pc_lat_df)
plt.show()

# Combining dataframes
MLset = pc_hold_df.join(pc_lat_df)
MLset['target'] = y

# %% Machine Learning
features = columns_lat + columns_hold
target = 'target'

X = MLset[features]
y = MLset[target]

# Hyperparametrization
param_neighbors = [x for x in range(1, 15)]
param_C = [(x/1000) for x in range(1, 50)]
param_depth = [x for x in range(1, 10)]

knn_best_param = 0
logreg_best_param = 0
tree_best_param = 0

knn_mcvs_best = 0
logreg_mcvs_best = 0
tree_mcvs_best = 0

for neighbor in param_neighbors:
    knn = KNeighborsClassifier(neighbor)
    mcvs_knn = np.mean(cross_val_score(knn, X, y, cv=10, scoring='accuracy'))
    if mcvs_knn > knn_mcvs_best:
        knn_best_param = neighbor
        knn_mcvs_best = mcvs_knn

for C_value in param_C:
    logreg = LogisticRegression(C=C_value)
    mcvs_logreg = np.mean(
                    cross_val_score(logreg, X, y, cv=10, scoring='accuracy'))
    if mcvs_logreg > logreg_mcvs_best:
        logreg_best_param = C_value
        logreg_mcvs_best = mcvs_logreg

for depth in param_depth:
    tree = DecisionTreeClassifier(max_depth=depth)
    mcvs_tree = np.mean(cross_val_score(tree, X, y, cv=10, scoring='accuracy'))
    if mcvs_tree > tree_mcvs_best:
        tree_best_param = depth
        tree_mcvs_best = mcvs_tree

knn = KNeighborsClassifier(knn_best_param)
logreg = LogisticRegression(logreg_best_param)
tree = DecisionTreeClassifier(tree_best_param)

y_pred_knn, y_test = model_pipeline(knn, X, y)
y_pred_logreg, y_test = model_pipeline(logreg, X, y)
y_pred_tree, y_test = model_pipeline(tree, X, y)


# %% Evaluation
cvs_knn, acc_knn, auc_knn, fpr_knn, tpr_knn, conf_knn, threshold_knn, probs_knn = model_evaluation(knn, X, y, y_pred_knn)
cvs_logreg, acc_logreg, auc_logreg, fpr_logreg, tpr_logreg, conf_logreg, threshold_logreg, probs_logreg = model_evaluation(logreg, X, y, y_pred_logreg)
cvs_tree, acc_tree, auc_tree, fpr_tree, tpr_tree, conf_tree, threshold_tree, probs_tree = model_evaluation(tree, X, y, y_pred_tree)

# %% Plotting
# Accuracy scores, tpr, fpr from Schrag et al (2002)
acc_all = (111+54)/200
acc_PD = (86+20)/123
acc_nonPD = (25+34)/77
fpr_all = 20/(20+54)
fpr_PD = 11/(11+20)
fpr_nonPD = 9/(9+34)
tpr_all = 111/(111+15)
tpr_PD = 86/(86+6)
tpr_nonPD = 25/(25+9)
xmargin = 0.02
ymargin = -0.02

plt.plot(fpr_knn, tpr_knn, label='knn')
plt.plot(fpr_logreg, tpr_logreg, label='logreg')
plt.plot(fpr_tree, tpr_tree, label='tree')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='no skill ref')
plt.plot([fpr_all, fpr_PD, fpr_nonPD], [tpr_all, tpr_PD, tpr_nonPD], 'ro')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Classifier Comparison')
plt.text(0.74, 0.5, 'AUC knn: {:.3f}'.format(auc_knn))
plt.text(0.74, 0.4, 'AUC logreg: {:.3f}'.format(auc_logreg))
plt.text(0.74, 0.6, 'AUC tree: {:.3f}'.format(auc_tree))
plt.text(fpr_all + xmargin, tpr_all + ymargin, 'All')
plt.text(fpr_PD + xmargin, tpr_PD + ymargin, 'Specialists')
plt.text(fpr_nonPD + xmargin, tpr_nonPD + ymargin, 'non-Specialists')
plt.legend()
plt.savefig('AUCclassifiers.png',bbox_inches='tight')
plt.show()
