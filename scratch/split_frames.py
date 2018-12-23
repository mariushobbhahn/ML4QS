import numpy as np
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt



set1_combined_frame = pd.read_csv('set1_combined_frame.csv', sep=',')
set2_combined_frame = pd.read_csv('set2_combined_frame.csv', set=',')

#collect only the columns with predictive value
set1_combined_frame = set1_combined_frame[['X', 'Y', 'Z', 'label', 'glucose_value']]
set2_combined_frame = set2_combined_frame[['X', 'Y', 'Z', 'label', 'glucose_value']]
all_combined_frame  = set1_combined_frame.append(set2_combined_frame, ignore_index=True)

set1_index_collection = [0, 61670, 70065, 105416, 105417, 105418, 133798, 155254, 161470, 161471, 179456, 179457, 219288, 221154, 229452, 234339, 236579, 236580, 236581, 263085, 305222, 369790, 371657, 386115, 412218, 428079, 432277, 435542, 451916, 456356, 459973, 461647, 461648, 461649, 461650, 461651, 490040, 517500, 571718, 580114, 604373, 609039, 613698, 617900, 620400, 628729, 638525, 737883, 744879, 751874, 758871, 772008, 778538, 783474, 811840, 852868, 855201, 865462, 871059, 871652, 871653, 899604, 936767, 950292, 1083183, 1092977, 1098486, 1110447, 1206703, 1206704, 1232982, 1386187, 1407645, 1409044, 1437758, 1448020, 1449886, 1455483, 1476940, 1483002, 1597739, 1631136, 1661094, 1683507, 1705261, 1745342, 1756070, 1765399, 1828644, 1914468, 1921459, 1947102, 1961260, 1988985, 2009503, 2025327, 2034656, 2052754, 2091765, 2120452, 2122318, 2131647, 2134446, 2332130, 2336793, 2340778, 2340779, 2368240, 2397768, 2440325, 2517628, 2527250, 2559432, 2566434, 2568766, 2583423, 2583424, 2583425, 2672068, 2686566, 2739750, 2759204, 2775545, 2796120, 2800318, 2912607, 2912608, 2936012, 2959801, 3084525, 3086858, 3090123, 3091085, 3091086, 3116329, 3125658, 3135453, 3137786, 3149912, 3186478, 3186479, 3222337, 3231666, 3243682, 3260008, 3265606, 3274002, 3274469, 3437633, 3452133, 3452134, 3501516, 3508047, 3517842, 3523440, 3526705, 3530437, 3567465, 3571663, 3696660, 3718114, 3770264, 3770265, 3794351, 3842663, 3848726, 3858055, 3862720, 3877646, 3884177, 3909825, 3911691, 4038629]

set2_index_collection = []

def add_one_hot_to_df(df):
    label_column = df.label
    unique_label = np.unique(label_column)


def get_index_collection(df_combi):
    index_collection = [0]
    for i in range(1, len(set1_combined_frame)):
        #show progress
        if(i % 100000 == 0):
            print(i)

        #collect the indices whenever the label changes
        if(df_combi.labels[i] != df_combi.labels[i - 1]):
            index_collection.append(i)

    return(index_collection)

print(get_index_collection(set2_combined_frame))

def seperate_frame_by_activity(df_combi, index_collection):

    #we split the dataframe at the given indices into smaller chunks and save them as np.arrays
    activity_list = []
    for j in range(1, len(index_collection)):
        activity_frame = set1_combined_frame.iloc[index_collection[j-1] + 1:index_collection[j]]
        activity_frame = activity_frame.values
        activity_list.append(activity_frame)

    activity_array = np.array(activity_list)

    return(activity_array)

set1_activity_array = seperate_frame_by_activity(set1_combined_frame, set1_index_collection)
