import numpy as np
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt

"""
set1_action_label = pd.read_csv('Set1_ActionLabel_cleaned.csv', sep=',')
set1_free_libre = pd.read_csv('Set1_FreeLibre.csv', sep=',')
set1_accelerometer = pd.read_csv('Set1_Accelerometer.csv', sep=',')
set1_combined_frame = pd.read_csv('set1_combined_frame.csv', sep=',')
"""

set2_action_label = pd.read_csv('Set2_ActionLabel_cleaned.csv', sep=',')
set2_free_libre = pd.read_csv('Set2_FreeLibre.csv', sep=',')
set2_accelerometer = pd.read_csv('Set2_Accelerometer.csv', sep=',')
set2_combined_frame = pd.read_csv('set2_combined_frame.csv', sep=',')


def aggregate_acc(df_combi, df_glucose, show_progress):

    #set up empty column
    sigma_x = np.zeros(len(df_glucose))
    sigma_y = np.zeros(len(df_glucose))
    sigma_z = np.zeros(len(df_glucose))
    mean_x = np.zeros(len(df_glucose))
    mean_y = np.zeros(len(df_glucose))
    mean_z = np.zeros(len(df_glucose))

    #running index
    last_index = 0
    #running counter
    glucose_counter = 0

    #iterate through the whole combined frame
    print("length of data: ", len(df_combi))

    for i in range(len(df_combi)-1):
        #iterate through labels
        if show_progress:
            if(i % 100000 == 0):
                print(i)

        if(df_combi.glucose_value.iloc[i] != df_combi.glucose_value.iloc[i+1] or i+1 == len(df_combi)):
            #update std and mean
            sigma_x[glucose_counter] = np.std(np.array(df_combi.X.iloc[last_index:i]))
            sigma_y[glucose_counter] = np.std(np.array(df_combi.Y.iloc[last_index:i]))
            sigma_z[glucose_counter] = np.std(np.array(df_combi.Z.iloc[last_index:i]))
            mean_x[glucose_counter] = np.mean(np.array(df_combi.X.iloc[last_index:i]))
            mean_y[glucose_counter] = np.mean(np.array(df_combi.Y.iloc[last_index:i]))
            mean_z[glucose_counter] = np.mean(np.array(df_combi.Z.iloc[last_index:i]))

            glucose_counter += 1
            last_index = i

    #create frame of columns
    df = pd.DataFrame({'sigma_x':sigma_x, 'sigma_y':sigma_y, 'sigma_z':sigma_z, 'mean_x':mean_x, 'mean_y':mean_y, 'mean_z':mean_z})

    return(df)

set2_glucose_agg = aggregate_acc(set2_combined_frame, set2_free_libre, show_progress=True)
#concatenate the two datasets
set2_glucose_acc = pd.concat([set2_free_libre, set2_glucose_agg], axis=1, ignore_index=False)

set2_glucose_acc.to_csv('set2_glucose_acc.csv', sep=',')
