import numpy as np
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split


set1_glucose_acc = pd.read_csv('set1_glucose_acc.csv', header = 0, sep=',')
set1_glucose_acc_selected = set1_glucose_acc[['glucose_value', 'sigma_x', 'sigma_y', 'sigma_z', 'mean_x', 'mean_y', 'mean_z']]
set2_glucose_acc = pd.read_csv('set2_glucose_acc.csv', header = 0, sep=',')
set2_glucose_acc_selected = set2_glucose_acc[['glucose_value', 'sigma_x', 'sigma_y', 'sigma_z', 'mean_x', 'mean_y', 'mean_z']]


def sliding_window(data, window_size, step_size, target_length, jump, acc_only):
    data_x = []
    data_Y = []
    #iterate over the dataset
    for i in np.arange(start=0, stop=len(data) - window_size - target_length - jump, step=step_size):
        #get the x_data:
        x_chunk = data.iloc[i:i + window_size]
        if target_length==1:
            y_target = data.glucose_value.iloc[i + window_size + jump]
        else:
            y_target = data.glucose_value.iloc[i + window_size + jump : i + window_size + target_length + jump]
        data_x.append(x_chunk.values)
        data_Y.append(y_target)

    if acc_only:
        data_x = np.delete(np.array(data_x), 0, axis=2)

    return(np.array(data_x), np.array(data_Y))


#save data in a file
def save_data(arr, filename=''):
    #open pickle file and save
    output = open(filename, 'wb')
    pickle.dump(arr, output)
    output.close()


#save all data with sliding window 1
data_x, data_Y = sliding_window(set1_glucose_acc_selected, window_size=10, step_size=1, target_length=1, acc_only=False, jump = 5)
x_train, x_test, Y_train, Y_test = train_test_split(data_x, data_Y, test_size=0, shuffle=False) #fake split
#print shapes of data:
print("shape of x_train: ", np.shape(x_train))
print("shape of Y_train: ", np.shape(Y_train))
print("shape of x_test: ", np.shape(x_test))
print("shape of Y_test: ", np.shape(Y_test))
data = np.array([x_train, Y_train, x_test, Y_test])
print("Y_train[0]: ", Y_train[0])
save_data(data, 'data/glucose_acc_set1_ws10_ss1_tl1_j5.pkl')
