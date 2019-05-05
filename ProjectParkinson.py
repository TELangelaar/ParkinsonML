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

from readprocess_tappy import process_user

import os

import warnings; warnings.filterwarnings('ignore')

#%% Directories
directory_tappy = "C:/Users/Desktop-TL/Documents/GitRepos/ParkinsonML/Tappy Data/"
directory_user = "C:/Users/Desktop-TL/Documents/GitRepos/ParkinsonML/Archived users/"

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
    user_tappy_data = process_user(directory_tappy, str(user_id), tappy_data)
    user_tappy_df.loc[user_id] = user_tappy_data

user_tappy_df = user_tappy_df.fillna(0)
user_tappy_df[user_tappy_df < 0] = 0  
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

#Males and Females for Parkinson: Stacked
ind = np.arange(2)
width = 0.5

Ptrue_M = user_df[user_df['Parkinsons']==1]
Ptrue_M = len(Ptrue_M.Male[Ptrue_M['Male']==1])
Ptrue_F = user_df[user_df['Parkinsons']==1]
Ptrue_F = len(Ptrue_F.Male[Ptrue_F['Male']==0])

Pfalse_M = user_df[user_df['Parkinsons']==0]
Pfalse_M = len(Pfalse_M.Male[Pfalse_M['Male']==1])
Pfalse_F = user_df[user_df['Parkinsons']==0]
Pfalse_F = len(Pfalse_F.Male[Pfalse_F['Male']==0])

men = (Pfalse_M,Ptrue_M)
women = (Pfalse_F,Ptrue_F)

with sns.color_palette(palette='bwr',n_colors=2): #Men in blue, women in pink
    g1 = plt.bar(ind,men,width)
    g2 = plt.bar(ind,women,width,bottom=men)
plt.xticks(ind, ('Healthy','Parkinsons'))
plt.ylim([0, 200])
plt.text(0, 65,cParkinsonsFalse,horizontalalignment='center',fontsize=20)
plt.text(1, 170,cParkinsonsTrue,horizontalalignment='center',fontsize=20)
plt.title('Distribution of gender among Parkinsons and Healthy')
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
combined_user_df = pd.concat([user_df, user_tappy_df], axis=1)

total_col_names = ['Hold time','Latency time','Flight time']
total_hand_names = ['LL','LR','LS','RL','RR','RS','SL','SR','SS']
N = len(total_hand_names)
for col in total_col_names:
    for col2 in total_hand_names:
        combined_user_df[col] = ((combined_user_df['LL_'+col]+
                                 combined_user_df['LR_'+col]+
                                 combined_user_df['LS_'+col]+
                                 combined_user_df['RL_'+col]+
                                 combined_user_df['RR_'+col]+
                                 combined_user_df['RS_'+col]+
                                 combined_user_df['SL_'+col]+
                                 combined_user_df['SR_'+col]+
                                 combined_user_df['SS_'+col])/N)
combined_user_df[combined_user_df['Hold time'] > 800] = 0
combined_user_df[combined_user_df['Latency time'] > 800] = 0
combined_user_df[combined_user_df['Flight time'] > 800] = 0

all_cols = ['BirthYear','Male','Parkinsons','Tremors','DiagnosisYear','Sided_Left','Sided_None','Sided_Right','UPDRS_1','UPDRS_2','UPDRS_3','UPDRS_4',"UPDRS_Don't know",'Impact','Levadopa','DA','MAOB','Other']+column_names
combined_user_df_copy = combined_user_df.melt(id_vars=all_cols,value_vars=['Hold time','Latency time','Flight time'],var_name='Kind',value_name='Time')

combined_user_df[combined_user_df['Impact'] == 0] = None
combined_user_df_copy[combined_user_df_copy['Impact'] == 0] = None
order = ['None','Mild','Medium','Severe']

#Boxplot, grouped by Parkinsons
sns.boxplot(x='Kind',y='Time',hue='Parkinsons',data=combined_user_df_copy)
plt.title('Mean times for Parkinsons')
plt.ylabel('Time (ms)')
plt.legend(loc='upper left')
plt.savefig('swarmtimes.png',bbox_inces='tight')
plt.show()

#Boxplot, grouped by Impact
sns.boxplot(x='Kind',y='Time',hue='Impact',hue_order=order,data=combined_user_df_copy)
plt.title('Mean times for Parkinsons')
plt.ylabel('Time (ms)')
plt.savefig('impacttimes.png',bbox_inces='tight')
plt.show()

#Scatterplot: Hold vs Latency
sns.scatterplot(x='Hold time',y='Latency time',hue='Impact',hue_order=order,data=combined_user_df)
plt.ylabel('Latency time (ms)')
plt.xlabel('Hold time (ms)')
plt.title('Mean times of every participant')
plt.savefig('scatterholdlatency.png',bbox_inces='tight')
plt.show()

#Scatterplot: Hold vs Flight
sns.scatterplot(x='Hold time',y='Flight time',hue='Impact',hue_order=order,data=combined_user_df)
plt.ylabel('Flight (ms)')
plt.xlabel('Hold time (ms)')
plt.title('Mean times of every participant')
plt.savefig('scatterholdflight.png',bbox_inces='tight')
plt.show()

#Scatterplot: Flight vs Latency
sns.scatterplot(x='Flight time',y='Latency time',hue='Impact',hue_order=order,data=combined_user_df)
plt.ylabel('Latency time (ms)')
plt.xlabel('Flight time (ms)')
plt.title('Mean times of every participant')
plt.savefig('scatterlatencyflight.png',bbox_inces='tight')
plt.show()
