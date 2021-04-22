import os
import pandas as pd
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
import torch as tf
from collections import Counter
from datetime import datetime
import math
from sklearn.metrics import confusion_matrix

def load_sensor_data_without_h(hname,lname,rname,sample_lens=36000):
    head_sensor_txt = np.genfromtxt(hname, delimiter=',', dtype=None, encoding=None)
    left_hand_sensor_txt = np.genfromtxt(lname, delimiter=',', dtype=None, encoding=None)
    right_hand_sensor_txt = np.genfromtxt(rname, delimiter=',', dtype=None, encoding=None)
    start_time = datetime.fromisoformat(head_sensor_txt[0][0]).timestamp()
    end_time = datetime.fromisoformat(head_sensor_txt[-1][0]).timestamp()
    time=end_time-start_time
    sample_timing=np.arange(0,time,time/sample_lens)
    row_len = 3*3  + 1 + 4
    head_data_length = len(head_sensor_txt)
    left_hand_data_length = len(left_hand_sensor_txt)
    right_hand_data_length = len(right_hand_sensor_txt) 
    total_len = 13*3
    head_sensor_array = np.zeros((head_data_length, row_len))
    left_hand_sensor_array = np.zeros((left_hand_data_length, row_len))
    right_hand_sensor_array = np.zeros((right_hand_data_length, row_len))
    total_sensor_array = np.zeros((len(sample_timing),total_len))
    
    for row_i, sensor_row in enumerate(head_sensor_txt):
        head_sensor_array[row_i, 0] = sensor_row[2]
        head_sensor_array[row_i, 1] = sensor_row[3]
        head_sensor_array[row_i, 2] = sensor_row[4]
        head_sensor_array[row_i, 3] = sensor_row[6]
        head_sensor_array[row_i, 4] = sensor_row[7]
        head_sensor_array[row_i, 5] = sensor_row[8]
        head_sensor_array[row_i, 6] = sensor_row[10]
        head_sensor_array[row_i, 7] = sensor_row[11]
        head_sensor_array[row_i, 8] = sensor_row[12]
        head_sensor_array[row_i, 9] = sensor_row[18]
        head_sensor_array[row_i, 10] = sensor_row[19]
        head_sensor_array[row_i, 11] = sensor_row[20]
        head_sensor_array[row_i, 12] = sensor_row[21]
        head_sensor_array[row_i, 13] = datetime.fromisoformat(sensor_row[0]).timestamp()-start_time
        
    for row_i, sensor_row in enumerate(left_hand_sensor_txt):
        left_hand_sensor_array[row_i, 0] = sensor_row[2]
        left_hand_sensor_array[row_i, 1] = sensor_row[3]
        left_hand_sensor_array[row_i, 2] = sensor_row[4]
        left_hand_sensor_array[row_i, 3] = sensor_row[6]
        left_hand_sensor_array[row_i, 4] = sensor_row[7]
        left_hand_sensor_array[row_i, 5] = sensor_row[8]
        left_hand_sensor_array[row_i, 6] = sensor_row[10]
        left_hand_sensor_array[row_i, 7] = sensor_row[11]
        left_hand_sensor_array[row_i, 8] = sensor_row[12]
        left_hand_sensor_array[row_i, 9] = sensor_row[18]
        left_hand_sensor_array[row_i, 10] = sensor_row[19]
        left_hand_sensor_array[row_i, 11] = sensor_row[20]
        left_hand_sensor_array[row_i, 12] = sensor_row[21]
        left_hand_sensor_array[row_i, 13] = datetime.fromisoformat(sensor_row[0]).timestamp()-start_time
    
    for row_i, sensor_row in enumerate(right_hand_sensor_txt):
        right_hand_sensor_array[row_i, 0] = sensor_row[2]
        right_hand_sensor_array[row_i, 1] = sensor_row[3]
        right_hand_sensor_array[row_i, 2] = sensor_row[4]
        right_hand_sensor_array[row_i, 3] = sensor_row[6]
        right_hand_sensor_array[row_i, 4] = sensor_row[7]
        right_hand_sensor_array[row_i, 5] = sensor_row[8]
        right_hand_sensor_array[row_i, 6] = sensor_row[10]
        right_hand_sensor_array[row_i, 7] = sensor_row[11]
        right_hand_sensor_array[row_i, 8] = sensor_row[12]
        right_hand_sensor_array[row_i, 9] = sensor_row[18]
        right_hand_sensor_array[row_i, 10] = sensor_row[19]
        right_hand_sensor_array[row_i, 11] = sensor_row[20]
        right_hand_sensor_array[row_i, 12] = sensor_row[21]
        right_hand_sensor_array[row_i, 13] = datetime.fromisoformat(sensor_row[0]).timestamp()-start_time
    
    for i, sample_time in enumerate(sample_timing):
        idx_h = (np.abs(head_sensor_array[:,-1] - sample_time)).argmin()
        idx_l = (np.abs(left_hand_sensor_array[:,-1] - sample_time)).argmin()
        idx_r = (np.abs(right_hand_sensor_array[:,-1] - sample_time)).argmin()
        total_sensor_array[i,0:13] = head_sensor_array[idx_h,0:13]
        total_sensor_array[i,13:26] = left_hand_sensor_array[idx_l,0:13]
        total_sensor_array[i,26:] = right_hand_sensor_array[idx_r,0:13]
    return total_sensor_array

def sample_sensor_data(input_data, window_sz = 128, sample_sz = 128):
    sensor_length = input_data.shape[0]
    feature_sz = input_data.shape[1]
    data_sz = 0
    for i in range(0, sensor_length-window_sz, sample_sz):
        data_sz = data_sz + 1
    all_sensor_data = np.zeros((data_sz, feature_sz, window_sz))
    cnt = 0
    for i in range(0, sensor_length-window_sz, sample_sz):
        sample = input_data[i:i + window_sz, :]
        sample = np.transpose(sample)
        all_sensor_data[cnt, :, :] = sample
        cnt = cnt + 1
    return all_sensor_data

def manual_lable_array_list(manual_lable_path, combine_list):
    label_array_list = {}
    manual_lable__csv = pd.read_csv(manual_lable_path)
    n_lable_permin = 2000
    for j in combine_list:
        label_array = []
        label_permin = manual_lable__csv.iloc[:,j-1]
        for i in label_permin:
            label = int(i)
            for k in range(n_lable_permin):
                label_array.append(label)
        label_array = np.asarray(label_array)
        label_array_list[j] = label_array
    np.save('../save_data/label_data/label_list.npy', label_array_list)

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def extract_sensor_data(combine_list):
    combine_sensor_data = {}
    save_counter = 0
    for j in combine_list:
    	save_counter += 1
    	head_sensor_name =  '../sensor_data/head_Video' + str(j) + '.txt'
    	left_hand_sensor_name = '../sensor_data/left_Video' + str(j) + '.txt'
    	right_hand_sensor_name = '../sensor_data/right_Video' + str(j) + '.txt'
    	combine_data = load_sensor_data_without_h(hname=head_sensor_name,lname=left_hand_sensor_name,rname=right_hand_sensor_name)
    	combine_sensor_data[j] = combine_data         
    	if save_counter%10 == 0:
            np.save('../save_data/original_data/combine_sensor_data_actor_{}.npy'.format(int(save_counter/10)), combine_sensor_data)
            combine_sensor_data = {}
            print('save to ../save_data/original_data/combine_sensor_data_actor_{}.npy'.format(int(save_counter/10)))

def get_sensor_data(combine_data_path):
    combine_data_list = sorted(os.listdir(combine_data_path))
    combine_data = {}
    for path in combine_data_list:
        combine_data_temp = np.load(combine_data_path+path, allow_pickle=True).item()
        combine_data.update(combine_data_temp)
    return combine_data

def sample_data(sensor_data, combine_list, window_sz=128, sample_sz=128):
    combine_sensor_sample_data = {}
    save_counter = 0
    for j in combine_list:
        save_counter += 1
        combine_data = sensor_data[j]
        combine_sample_data = sample_sensor_data(combine_data,window_sz = window_sz, sample_sz = sample_sz)
        combine_sensor_sample_data[j] = combine_sample_data
        if save_counter%10 == 0:
            np.save('../save_data/sample_data/combine_sensor_sample_data_actor_{}.npy'.format(int(save_counter/10)), combine_sensor_sample_data)
            combine_sensor_sample_data = {}
            print('save to ../save_data/sample_data/combine_sensor_sample_data_actor_{}.npy'.format(int(save_counter/10)))

def sample_label(label_list, combine_list, window_sz=128, sample_sz=128):
    combine_sensor_sample_label = {}
    save_counter = 0
    for j in combine_list:
        save_counter += 1
        label_data=label_list[j]
        all_label = np.zeros((int(label_data.shape[0]/sample_sz), 1), dtype=int)
        cnt = 0
        for k in range(0, label_data.shape[0]-window_sz, sample_sz):
            cur_label_array = label_data[k:k+window_sz]
            all_label[cnt] = most_frequent(cur_label_array)
            cnt = cnt + 1
        combine_sensor_sample_label[j] = all_label
        if save_counter%10 == 0:
            np.save('../save_data/sample_label/combine_sensor_sample_label_actor_{}.npy'.format(int(save_counter/10)), combine_sensor_sample_label)
            combine_sensor_sample_data = {}
            print('save to ../save_data/sample_label/combine_sensor_sample_label_actor_{}.npy'.format(int(save_counter/10)))
            
def combine_to_one(data, co_list):
    for i in range(len(co_list)):
        if i == 0:
            final_data = data[co_list[0]]
        else:
            final_data = np.concatenate((final_data, data[co_list[i]]),axis=0)
    return final_data
    






