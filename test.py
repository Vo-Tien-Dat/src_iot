import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D

from sklearn.model_selection import train_test_split

file_paths_test = [
     '../data/part-00007-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
]

ORDINARY_LABEL_NAME = 'BenignTraffic'

col_names = [
    'flow_duration',
    'Header_Length',
    'Protocol_Type',
    'Duration',
    'Rate',
    'Srate',
    'Drate',
    'fin_flag_number',
    'syn_flag_number',
    'rst_flag_number',
    'psh_flag_number',
    'ack_flag_number',
    'ece_flag_number',
    'cwr_flag_number',
    'ack_count',
    'syn_count',
    'fin_count',
    'urg_count',
    'rst_count',
    'HTTP',
    'HTTPS',
    'DNS',
    'Telnet',
    'SMTP',
    'SSH',
    'IRC',
    'TCP',
    'UDP',
    'DHCP',
    'ARP',
    'ICMP',
    'IPv',
    'LLC',
    'tot_sum',
    'Min',
    'Max',
    'AVG',
    'Std',
    'Tot_size',
    'IAT',
    'Number',
    'Magnitue',
    'Radius',
    'Covariance',
    'Variance',
    'Weight',
    'label']

CONST_DROPPED_LABEL = [
    'HTTP', 
    'HTTPS', 
    'DNS', 
    'Telnet', 
    'SMTP', 
    'SSH', 
    'IRC', 
    'TCP', 
    'DHCP', 
    'ICMP', 
    'IPv', 
    'LLC']

def merge_data_from_multiple_file(file_paths = [], col_names = []):
    data = pd.DataFrame()

    if len(file_paths) == 0:
        raise ValueError('Vui lòng thêm đường dẫn')
    
    for file_path in file_paths: 
        sub_data = pd.read_csv(file_path, names = col_names)

        if len(sub_data) == 0: 
            raise ValueError('Dữ liệu trong file ${}'.format(file_path))
        
        if len(data) == 0: 
            data = pd.concat([data, sub_data])
            continue

        if len(data.columns) != len(data.columns): 
            raise  ValueError('số lượng cột không khớp ! Vui lòng kiểm tra lại số lượng cột')
        
        data = pd.concat([data, sub_data])

    return data

data_test = merge_data_from_multiple_file(file_paths=file_paths_test, col_names=col_names)


def delete_field_of_data(data, fields = []):
    data = data.copy() 
    new_data = data.drop(fields, axis = 1)
    return new_data

def split_label_two_classes(data): 
    new_data = data.copy()

    def group_two_label(label): 
        if label == ORDINARY_LABEL_NAME: 
            return 0 
        return 1

    new_data['two_label'] = new_data['label'].apply(group_two_label)
    return new_data


new_data_test = split_label_two_classes(data_test)

new_data_test = delete_field_of_data(new_data_test, fields= CONST_DROPPED_LABEL)

exec_columns = [['label', 'two_label']]

new_data_test['label'].value_counts().to_frame()



