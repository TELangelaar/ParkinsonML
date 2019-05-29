# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:17:25 2019

@author: Desktop-TL
Functions definitions to read and process user and tappy data from Parkinsons
"""
import pandas as pd
import numpy as np

def read_tappy(directory, file_name):
    df = pd.read_csv(
        directory + file_name,
        delimiter = '\t',
        index_col = False,
        names = ['UserKey', 'Date', 'Timestamp', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time']
    )
#    df.set_index('UserKey', inplace=True)
#    df['Date'] = pd.to_datetime(df['Date'], errors='coerce',
#                                format='%y%M%d').dt.date
#    df['Timestamp'] = pd.to_timedelta(df['Timestamp'], errors='coerce',
#                                      unit='milli')

    # converting time data to numeric
    for column in ['Hold time', 'Latency time', 'Flight time']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
#    df = df.dropna(axis=0)

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
    
    # making features for Hold Time
    temp_array = np.array([])
    
    temp_array = np.append(temp_array, df[df['Hand'] == 'L']['Hold time'].mean())
    temp_array = np.append(temp_array, df[df['Hand'] == 'L']['Hold time'].std())
    temp_array = np.append(temp_array, df[df['Hand'] == 'L']['Hold time'].kurtosis())
    temp_array = np.append(temp_array, df[df['Hand'] == 'L']['Hold time'].skew())
    
    temp_array = np.append(temp_array, df[df['Hand'] == 'L']['Hold time'].mean())
    temp_array = np.append(temp_array, df[df['Hand'] == 'R']['Hold time'].std())
    temp_array = np.append(temp_array, df[df['Hand'] == 'R']['Hold time'].kurtosis())
    temp_array = np.append(temp_array, df[df['Hand'] == 'R']['Hold time'].skew())

    temp_array = np.append(temp_array, df[df['Hand'] == 'L']['Hold time'].mean() - df[df['Hand'] == 'R']['Hold time'].mean())
    
    # making features for Latency Time
    temp_array = np.append(temp_array, df[df['Direction'] == 'LR']['Latency time'].mean())
    temp_array = np.append(temp_array, df[df['Direction'] == 'LR']['Latency time'].std())
    temp_array = np.append(temp_array, df[df['Direction'] == 'LR']['Latency time'].kurtosis())
    temp_array = np.append(temp_array, df[df['Direction'] == 'LR']['Latency time'].skew())
    
    temp_array = np.append(temp_array, df[df['Direction'] == 'RL']['Latency time'].mean())
    temp_array = np.append(temp_array, df[df['Direction'] == 'RL']['Latency time'].std())
    temp_array = np.append(temp_array, df[df['Direction'] == 'RL']['Latency time'].kurtosis())
    temp_array = np.append(temp_array, df[df['Direction'] == 'RL']['Latency time'].skew())
    
    temp_array = np.append(temp_array, df[df['Direction'] == 'LL']['Latency time'].mean())
    temp_array = np.append(temp_array, df[df['Direction'] == 'LL']['Latency time'].std())
    temp_array = np.append(temp_array, df[df['Direction'] == 'LL']['Latency time'].kurtosis())
    temp_array = np.append(temp_array, df[df['Direction'] == 'LL']['Latency time'].skew())
    
    temp_array = np.append(temp_array, df[df['Direction'] == 'RR']['Latency time'].mean())
    temp_array = np.append(temp_array, df[df['Direction'] == 'RR']['Latency time'].std())
    temp_array = np.append(temp_array, df[df['Direction'] == 'RR']['Latency time'].kurtosis())
    temp_array = np.append(temp_array, df[df['Direction'] == 'RR']['Latency time'].skew())
    
    temp_array = np.append(temp_array, df[df['Direction'] == 'LR']['Latency time'].mean() - df[df['Direction'] == 'RL']['Latency time'].mean())
    temp_array = np.append(temp_array, df[df['Direction'] == 'LL']['Latency time'].mean() - df[df['Direction'] == 'RR']['Latency time'].mean())

    return temp_array # returning a numppy array

def process_user(directory_tappy, user_id, filenames):
    running_user_data = np.array([])

    for filename in filenames:
        if user_id in filename:
            running_user_data = np.append(running_user_data, read_tappy(directory_tappy,filename))
    
    running_user_data = np.reshape(running_user_data, (-1, 27))
    return np.mean(running_user_data, axis=0)

#%%
#tappy_names = ['L_Hand_mean', 'L_Hand_std', 'L_Hand_kurt', 'L_Hand_skew',
#               'R_Hand_mean', 'R_Hand_std', 'R_Hand_kurt', 'R_Hand_skew', 
#               'diff_Hand_mean',
#               'LR_mean', 'LR_std', 'LR_kurt', 'LR_skew',
#               'RL_mean', 'RL_std', 'RL_kurt', 'RL_skew',
#               'LL_mean', 'LL_std', 'LL_kurt', 'LL_skew',
#               'RR_mean', 'RR_std', 'RR_kurt', 'RR_skew',
#               'diff_opposite_mean', 'diff_same_mean']
#
#user_tappy_df = pd.DataFrame(columns=tappy_names)
#directory_tappy = "Tappy Data/"
#user_id = '0EA27ICBLF'
#filenames = ['0EA27ICBLF_1607.txt','0EA27ICBLF_1608.txt']
#
#
#user_tappy_data = process_user(directory_tappy, str(user_id), filenames)
#user_tappy_df.loc[user_id] = user_tappy_data
