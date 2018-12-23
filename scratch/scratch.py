import numpy as np
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime




#datetime.timedelta converted to hh:mm:ss format
def hours_minutes_seconds(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    result = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    return(result)

#time stamp in human readable format to local time stamp
def date_time_to_local_time(timestamp):
    return(int(time.mktime(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))) - time.timezone)

#local time to human readable format
def local_time_to_date_time(local_time):
    return(time.strftime('%d.%m.%Y %H:%M:%S', time.localtime(int(local_time/1000))))


#transformation from hh:mm:ss to seconds
def get_msec(time_str):
    h, m, s = time_str.split(':')
    sec = int(h) * 3600 + int(m) * 60 + int(s)
    return(sec*1000)



#add durations in msec
def add_durations_msec(df):

    durations_msec = np.append(np.diff(df.timestamp), np.array([0]))
    df['duration_msec'] = durations_msec

    return(df)


#test transformation:
#print(timestamp_to_local_time('2018-01-20 07:31:00'))

def add_timestamps(df):
    timestamps = []
    for i in range(len(df)):
        timestamp_i = date_time_to_local_time(df.date_time[i])
        timestamp_i *= 1000
        #print("local_time: ", local_time)
        timestamps.append(timestamp_i)

    df['timestamp'] = timestamps

    return(df)



"""
We take the glucose measurements and take all measurements that are during the periods of sleep
"""

def get_glucose_level(df):
    list_selected = []
    for i in range(len(df)):
        for j in range(len(set1_free_libre)):
            condition = set1_free_libre.timestamp.iloc[j] > df.timestamp.iloc[i] \
                and   set1_free_libre.timestamp.iloc[j] < df.timestamp.iloc[i] + df.duration_msec.iloc[i]
            if(condition):
                list_selected.append([set1_free_libre.timestamp.iloc[j], set1_free_libre.glucose_value.iloc[j]])
    return(pd.DataFrame(list_selected))


#throw all the data into one big DataFrame


def add_label_to_frame(df_acc, df_label, show_progress=False):
    #set up empty columns
    combined_labels = np.array([''] * len(df_acc), dtype = object)
    #iterate through the whole combined frame
    print("length of data: ", len(df_acc))

    label_counter = 0
    for i in range(len(df_acc)):
        #iterate through labels
        if show_progress:
            if(i % 100000 == 0):
                print(i)

        if(df_acc.timestamp.iloc[i] < df_label.timestamp.iloc[label_counter] + df_label.duration_msec.iloc[label_counter]):
            combined_labels[i] = df_label.action.iloc[label_counter]
        elif(label_counter == len(df_label) - 1):
            break
        else:
            label_counter += 1
            combined_labels[i] = df_label.action.iloc[label_counter]

    return(combined_labels)

def add_glucose_level_to_frame(df_acc, df_glucose, show_progress):
    #set up empty column
    combined_glucose = np.zeros(len(df_acc))
    #iterate through the whole combined frame
    print("length of data: ", len(df_acc))

    glucose_counter = 0
    for i in range(len(df_acc)):
        #iterate through labels
        if show_progress:
            if(i % 100000 == 0):
                print(i)

        if(df_acc.timestamp.iloc[i] < df_glucose.timestamp.iloc[glucose_counter] + df_glucose.duration_msec.iloc[glucose_counter]):
            combined_glucose[i] = df_glucose.glucose_value.iloc[glucose_counter]
        elif(glucose_counter == len(df_glucose) - 1):
            break
        else:
            glucose_counter += 1
            combined_glucose[i] = df_glucose.glucose_value.iloc[glucose_counter]

    return(combined_glucose)

def show_label_counts(df_action_label):
    a, b = np.unique(df_action_label.action, return_counts=True)

    print("different possible classes of activities: ", a, b)


"""
#since not all frames are sorted for timestamps we sort the frame by this attribute
set1_accelerometer = set1_accelerometer.sort_values(by= ['timestamp'])
set1_free_libre = set1_free_libre.sort_values(by= ['timestamp'])
set1_action_label = set1_action_label.sort_values(by= ['timestamp'])

#and now check again whether they match
print("first timestamp accelerometer: ", set1_accelerometer.head(n=1))
print("first timestamp glucose: ", set1_free_libre.head(n=1))
print("first timestamp action label: ", set1_action_label.head(n=1))



#now we can put all data in one big frame
combined_frame = set1_accelerometer

set1_action_label = pd.read_csv('Set1_ActionLabel_cleaned.csv', sep=',')
set1_free_libre = pd.read_csv('Set1_FreeLibre.csv', sep=',')
set1_accelerometer = pd.read_csv('Set1_Accelerometer.csv', sep=',')

set1_action_label = add_timestamps(set1_action_label)
set1_action_label = add_durations_msec(set1_action_label)
set1_free_libre = add_durations_msec(set1_free_libre)

set1_free_libre.to_csv('Set1_FreeLibre.csv', sep=',')

set1_action_label = add_timestamps(set1_action_label)

#select only relevant columns of the action label set: action, local_time, duration
set1_action_label_selected = set1_action_label[['action', 'timestamp', 'duration_msec']]

show_label_counts(set1_action_label_selected)

set1_action_label_selected.to_csv('set1_action_label_selected.csv', sep=',')

combined_frame = set1_accelerometer
combined_frame['label'] = add_label_to_frame(set1_accelerometer, set1_action_label, show_progress=True)
combined_frame['glucose_value'] = add_glucose_level_to_frame(set1_accelerometer, set1_free_libre, show_progress=True)

combined_frame.to_csv('set1_combined_frame.csv', sep=',')
"""


set2_action_label = pd.read_csv('Set2_ActionLabel_cleaned.csv', sep=',')
set2_free_libre = pd.read_csv('Set2_FreeLibre.csv', sep=',')
set2_accelerometer = pd.read_csv('Set2_Accelerometer.csv', sep=',')

set2_action_label = add_timestamps(set2_action_label)
set2_action_label = add_durations_msec(set2_action_label)
set2_free_libre = add_durations_msec(set2_free_libre)

set2_accelerometer = set2_accelerometer.sort_values(by= ['timestamp'])
set2_free_libre = set2_free_libre.sort_values(by= ['timestamp'])
set2_action_label = set2_action_label.sort_values(by= ['timestamp'])

set2_free_libre.to_csv('Set2_FreeLibre.csv', sep=',')

set2_action_label = add_timestamps(set2_action_label)

#select only relevant columns of the action label set: action, local_time, duration
set2_action_label_selected = set2_action_label[['action', 'timestamp', 'duration_msec']]

show_label_counts(set2_action_label_selected)

set2_action_label_selected.to_csv('set2_action_label_selected.csv', sep=',')

combined_frame = set2_accelerometer
combined_frame['label'] = add_label_to_frame(set2_accelerometer, set2_action_label, show_progress=True)
combined_frame['glucose_value'] = add_glucose_level_to_frame(set2_accelerometer, set2_free_libre, show_progress=True)

combined_frame.to_csv('set2_combined_frame.csv', sep=',')
