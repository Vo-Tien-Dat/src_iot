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

file_paths  = [
    '../data/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    '../data/part-00001-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    '../data/part-00002-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv', 
    '../data/part-00003-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    '../data/part-00004-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
]

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

data = merge_data_from_multiple_file(file_paths=file_paths, col_names=col_names)

## Nhãn bình thường
ORDINARY_LABEL_NAME = 'BenignTraffic'

## Gồm có 7 nhãn tấn công

DDOS_LABEL = [
    'DDoS-ICMP_Flood', 
    'DDoS-UDP_Flood', 
    'DDoS-TCP_Flood', 
    'DDoS-PSHACK_Flood', 
    'DDoS-SYN_Flood', 
    'DDoS-RSTFINFlood', 
    'DDoS-SynonymousIP_Flood', 
    'DDoS-ICMP_Fragmentation', 
    'DDoS-ACK_Fragmentation', 
    'DDoS-UDP_Fragmentation', 
    'DDoS-HTTP_Flood', 
    'DDoS-SlowLoris']

DOS_LABEL = [
    'DoS-UDP_Flood', 
    'DoS-TCP_Flood', 
    'DoS-SYN_Flood', 
    'DoS-HTTP_Flood']

MIRAI_LABEL = [
    'Mirai-greeth_flood', 
    'Mirai-udpplain', 
    'Mirai-greip_flood']

SPOOFING_LABEL = [
    'MITM-ArpSpoofing', 
    'DNS_Spoofing']

RECON_LABEL = [
    'Recon-HostDiscovery', 
    'Recon-OSScan', 
    'Recon-PortScan', 
    'Recon-PingSweep', 
    'VulnerabilityScan']

WEB_LABEL = [
    'SqlInjection', 
    'BrowserHijacking', 
    'CommandInjection', 
    'Backdoor_Malware', 
    'XSS', 
    'Uploading_Attack']

BRUTE_FORCE_LABEL = [
    'DictionaryBruteForce']

PROTOCOL_FIELDS = [
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
    'LLC'
]

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

def delete_field_of_data(data, fields = []):
    data = data.copy() 
    new_data = data.drop(fields, axis = 1)
    return new_data

data = delete_field_of_data(data, fields= CONST_DROPPED_LABEL)

## Tách tập dữ liệu thành 2 phần gồm có dữ liệu bình thường và dữ liệu tấn công
def split_label_two_classes(data): 
    new_data = data.copy()

    def group_two_label(label): 
        if label == ORDINARY_LABEL_NAME: 
            return 0 
        return 1

    new_data['two_label'] = new_data['label'].apply(group_two_label)
    return new_data

## Tách tập dữ liệu thành 7 nhóm tấn công gồm có DDoS, DoS, Mirai, Spoofing, Recon, Web, Bruce Force, Web
def split_label_eight_classes(data): 
    new_data = data.copy()
    return new_data


new_data = split_label_two_classes(data)

### Training model LSTM
def LSTM_model(data):
    data = data.copy()
    data = data.drop(['label'], axis = 1)
    X_data, Y_data = data.iloc[:, 1: -1], data.iloc[:, -1]

    
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2)


    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


    # Y_train = np.reshape(Y_train, (Y_train.shape[0], 1, Y_train.shape[1]))
    # Y_test = np.reshape(Y_test, (Y_test.shape[0], 1, Y_test.shape[1]))

    model = Sequential()
    model.add(LSTM(units = 64, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer = "adam", metrics = ['accuracy'], loss = 'binary_crossentropy')
    model.summary()
    model.fit(X_train, Y_train, epochs = 16, batch_size = 32, validation_data = (X_test, Y_test))
    model.save("two_label_predictions.h5")


LSTM_model(new_data)