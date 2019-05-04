"""
Created on Sat May  4 14:01:58 2019

@author: Thijme Langelaar
Time:
    04-05: 1358-1734
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os, gc

import warnings; warnings.filterwarnings('ignore')

#%% Directories
directory_tappy = "E:/Documents/Studie/Hoorcollege's/Big Data & Data Science/ProjectParkinson/Tappy Data/"
directory_user = "E:/Documents/Studie/Hoorcollege's/Big Data & Data Science/ProjectParkinson/Archived users/"

#%% Function Definitions
def read_tappy(directory, file_name):
    df = pd.read_csv(
        directory + file_name,
        delimiter = '\t',
        index_col = False,
        names = ['UserKey', 'Date', 'Timestamp', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time']
    )

    df = df.drop('UserKey', axis=1)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%y%M%d').dt.date

    # converting time data to numeric
    #print(df[df['Hold time'] == '0105.0EA27ICBLF']) # for 0EA27ICBLF_1607.txt
    for column in ['Hold time', 'Latency time', 'Flight time']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(axis=0)

    # cleaning data in Hand
    df = df[
        (df['Hand'] == 'L') |
        (df['Hand'] == 'R') |
        (df['Hand'] == 'S')
    ]

    # cleaning data in Direction
    df = df[
        (df['Direction'] == 'LL') |
        (df['Direction'] == 'LR') |
        (df['Direction'] == 'LS') |
        (df['Direction'] == 'RL') |
        (df['Direction'] == 'RR') |
        (df['Direction'] == 'RS') |
        (df['Direction'] == 'SL') |
        (df['Direction'] == 'SR') |
        (df['Direction'] == 'SS')
    ]

    direction_group_df = df.groupby('Direction').mean()
    del df; gc.collect()
    direction_group_df = direction_group_df.reindex(['LL', 'LR', 'LS', 'RL', 'RR', 'RS', 'SL', 'SR', 'SS'])
    direction_group_df = direction_group_df.sort_index() # to ensure correct order of data
    
    return direction_group_df.values.flatten() # returning a numppy array


def process_user(user_id, filenames):
    running_user_data = np.array([])

    for filename in filenames:
        if user_id in filename:
            running_user_data = np.append(running_user_data, read_tappy(directory_tappy,filename))
    
    running_user_data = np.reshape(running_user_data, (-1, 27))
    return np.nanmean(running_user_data, axis=0) # ignoring NaNs while calculating the mean

#%% Data Import
user_data = os.listdir(directory_user)
tappy_data = os.listdir(directory_tappy)

ID_data = [x[5:15] for x in user_data] #Get user ID from file name
ID_tappy = [x[0:10] for x in tappy_data]   

users_actual = [value for value in ID_data if value in ID_tappy] #If the user has no Tappy data, we dont need that user

#Import User Data
columns = ['BirthYear', 'Gender', 'Parkinsons', 'Tremors', 'DiagnosisYear',
    'Sided', 'UPDRS', 'Impact', 'Levadopa', 'DA', 'MAOB', 'Other']
user_df = pd.DataFrame(columns=columns)
for userid in users_actual:
    f = open("E:/Documents/Studie/Hoorcollege's/Big Data & Data Science/ProjectParkinson/Archived users/"+'User_'+userid+'.txt')
    user_df.loc[userid] = [line.split(': ')[1][: -1] for line in f.readlines()]
    f.close()

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
    (user_df['Impact'] != 'Severe'), 'Impact'] = 'None'

to_dummy_column_indices = ['Sided', 'UPDRS'] # columns to be one-hot encoded
for column in to_dummy_column_indices:
    user_df = pd.concat([
        user_df.iloc[:, : user_df.columns.get_loc(column)],
        pd.get_dummies(user_df[column], prefix=str(column)),
        user_df.iloc[:, user_df.columns.get_loc(column) + 1 :]
    ], axis=1)


#Import Tappy Data                    
column_names = [first_hand + second_hand + '_' + time for first_hand in ['L', 'R', 'S'] for second_hand in ['L', 'R', 'S'] for time in ['Hold time', 'Latency time', 'Flight time']]

user_tappy_df = pd.DataFrame(columns=column_names)

for user_id in user_df.index:
    user_tappy_data = process_user(str(user_id), tappy_data)
    user_tappy_df.loc[user_id] = user_tappy_data
    
#%% Data Visualisation: User Data
N = len(users_actual)

#Missing Data
missing_data = user_df.isnull().sum()
g = sns.barplot([missing_data.index[0],missing_data.index[4]],[missing_data.BirthYear,missing_data.DiagnosisYear],palette='bwr_r')
cBirthyear = round(missing_data.BirthYear/(N/100),2)
cBirthyear = str(cBirthyear) + '%'
cDiagnosisyear = round(missing_data.DiagnosisYear/(N/100),2)
cDiagnosisyear = str(cDiagnosisyear) + '%'
g.text(0,40,cBirthyear,horizontalalignment='center',fontsize=20)
g.text(1,72,cDiagnosisyear,horizontalalignment='center',fontsize=20)
plt.ylim([0,85])
plt.title('Missing Data, N = '+str(len(users_actual)))
plt.savefig('missingdata.png',bbox_inches='tight')
plt.show()

#People with Parkinsons
g = sns.countplot(user_df.Parkinsons,palette='bwr_r')
cParkinsonsFalse = round(user_df.Parkinsons[user_df['Parkinsons']==0].count()/(N/100),2)
cParkinsonsFalse = str(cParkinsonsFalse) + '%'
cParkinsonsTrue = round(user_df.Parkinsons[user_df['Parkinsons']==1].count()/(N/100),2)
cParkinsonsTrue = str(cParkinsonsTrue) + '%'
g.text(0,65,cParkinsonsFalse,horizontalalignment='center',fontsize=20)
g.text(1,170,cParkinsonsTrue,horizontalalignment='center',fontsize=20)
plt.ylim([0,190])
plt.title('Number of people with Parkinsons')
plt.savefig('countparkinsons.png',bbox_inches='tight')
plt.show()

#Males and Females
g = sns.countplot(user_df.Male,palette='bwr_r')
cMale = round(user_df.Male[user_df['Male']==1].count()/(N/100),2)
cMale = str(cMale) + '%'
cFemale = round(user_df.Male[user_df['Male']==0].count()/(N/100),2)
cFemale = str(cFemale) + '%'
g.text(0,105,cFemale,horizontalalignment='center',fontsize=20)
g.text(1,120,cMale,horizontalalignment='center',fontsize=20)
plt.ylim([0,140])
plt.title('Number of males and females')
plt.savefig('countgender.png',bbox_inches='tight')
plt.show()

#Males and Females for Parkinson
g = sns.countplot(x='Parkinsons',hue='Male',data=user_df,palette='bwr_r')
cPM_false = user_df[user_df['Parkinsons']==0]
cPM_false = str(round(cPM_false.Male[cPM_false['Male']==1].count()/(cPM_false.Parkinsons.count()/100)))+'%'
cPM_true = user_df[user_df['Parkinsons']==1]
cPM_true = str(round(cPM_true.Male[cPM_true['Male']==1].count()/(cPM_true.Parkinsons.count()/100)))+'%'
cPF_false = user_df[user_df['Parkinsons']==0]
cPF_false = str(round(cPF_false.Male[cPF_false['Male']==0].count()/(cPF_false.Parkinsons.count()/100)))+'%' 
cPF_true = user_df[user_df['Parkinsons']==1]
cPF_true = str(round(cPF_true.Male[cPF_true['Male']==0].count()/(cPF_true.Parkinsons.count()/100)))+'%'
g.text(-0.35,28,cPF_false,fontsize=15)
g.text(0.05,35,cPM_false,fontsize=15)
g.text(0.65,80,cPF_true,fontsize=15)
g.text(1.05,88,cPM_true,fontsize=15)
plt.ylim([0, 110])
plt.title('Distribution of gender for Parkinsons')
plt.savefig('parkinsonscountgender.png',bbox_inches='tight')
plt.show()

#Tremors for Parkinsons
g = sns.countplot(x='Parkinsons',hue='Tremors',data=user_df,palette='bwr_r')
plt.title('Distribution of tremors for Parkinsons')
plt.savefig('parkinsonscounttremors.png',bbox_inches='tight')
plt.show()

#Impact
g = sns.countplot(x='Impact',hue='Parkinsons',order=['None','Mild','Medium','Severe'],data=user_df,palette='bwr_r')
cNone = str(round(user_df.Impact[user_df['Impact']=='None'].count()/(N/100),2)) + '%'
cMild = str(round(user_df.Impact[user_df['Impact']=='Mild'].count()/(N/100),2)) + '%'
cMedium = str(round(user_df.Impact[user_df['Impact']=='Medium'].count()/(N/100),2)) + '%'
cSevere = str(round(user_df.Impact[user_df['Impact']=='Severe'].count()/(N/100),2)) + '%'
plt.title('Severity of Parkinsons')
g.text(-0.4,55,cNone,fontsize=12)
g.text(1,67,cMild,fontsize=12)
g.text(2,72,cMedium,fontsize=12)
g.text(3,27,cSevere,fontsize=12)
plt.ylim([0,90])
plt.savefig('countimpact.png',bbox_inches='tight')
plt.show()

#%% Data Visualisation: Tappy Data