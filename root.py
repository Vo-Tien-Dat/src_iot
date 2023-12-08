import os
import csv
import logging
from datetime import datetime

# Graphics
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# Data Proccessing
import pandas as pd
import numpy as np
import tensorflow as tf

## Visualization and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import sys
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

## Using k-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
import joblib
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import MinMaxScaler

from constant import * 


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

data = merge_data_from_multiple_file(file_paths=CONST_FILE_PATHS, col_names= CONST_FIELDS)

for cate_field in CONST_CATEGORICAL_FIELDS:
    data[cate_field] = data[cate_field].astype('category')


def display_wheel_chart(data, group_labels = [], label_text = [], label_name = 'label'):
    data = data.copy()
    label_values = []

    for group_label in group_labels:
      label_value = len(data[data[label_name].isin(group_label)])
      label_values.append(label_value)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')

    ax.pie(label_values, labels = label_text, autopct = '%1.2f%%')
    plt.show()


# display_wheel_chart(
#     data,
#     group_labels = [
#         CONST_DDOS_LABEL,
#         CONST_DOS_LABEL,
#         CONST_MIRAI_LABEL,
#         CONST_SPOOFING_LABEL,
#         CONST_RECON_LABEL,
#         CONST_WEB_LABEL,
#         CONST_BRUTE_FORCE_LABEL
#     ],
#     label_text =  [
#         'DDoS',
#         'DoS',
#         'Mirai',
#         'Spoofing',
#         'Recon',
#         'Web',
#         'Bruce Force'
#     ])

## Mô hình để phân biệt lớp tấn công và lớp bình thường
def get_LSTM_model(time_steps, features):
    model = Sequential()
    model.add(LSTM(units = 46,return_sequences= True, input_shape = (time_steps, features)))
    model.add(Dropout(0.1))
    model.add(LSTM(units = 46))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer = "adam", metrics = ['accuracy'], loss = 'binary_crossentropy')
    return model


## hiện confusion matrix
def log_confusion_maxtrix(model, X_test, Y_test, title):
    Y_pred = model.predict(X_test)
    Y_pred = np.round(Y_pred).flatten()

    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
    plt.title(title)
    plt.xlabel("Dữ liệu dự đoán")
    plt.ylabel("Dữ liệu thực tế")
    plt.show()

## Thực hiện cân bằng dữ liệu
def perform_imbalance_dataset(X, Y, fit_algorithm):
    X_algo, Y_algo = fit_algorithm.fit_resample(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X_algo, Y_algo, test_size = 0.2)
    return [X_train, X_test, Y_train, Y_test]

def split_group_label(data, group_labels = [], label_name = 'label'):
    sub_data_labels = []

    for group_label in group_labels:
        sub_data_label = data[data[label_name].isin(group_label)]
        sub_data_labels.append(sub_data_label)

    return sub_data_labels

def perform_LSTM_model(X, Y, imbalance_algo):
    X = X.copy()
    Y = Y.copy()
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)
    display_wheel_chart(pd.concat([X, Y], axis = 1), [[0], [1]], ['benign', 'attack'])
    X, Y = imbalance_algo.fit_resample(X, Y)
    merge_two_classes = pd.concat([X, Y], axis = 1)
    display_wheel_chart(merge_two_classes, [[0], [1]], ['benign', 'attack'])

    time_steps = 1
    features = len(X.columns)

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        X_train = X_train.values
        X_test = X_test.values

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test =  np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model = get_LSTM_model(time_steps, features)
        check_point_the_best_model = ModelCheckpoint(
                                filepath=CONST_NAME_MODEL,
                                monitor='accuracy',
                                verbose=1,
                                save_best_only=True,
                                mode='max')

        callbacks = [check_point_the_best_model]
        model.fit(X_train, Y_train, epochs = 10, batch_size = 128, validation_data = (X_test, Y_test),  callbacks=callbacks)
        model.summary()
        log_confusion_maxtrix(model, X_test, Y_test, 'Đồ thị')
        model.save(CONST_NAME_MODEL)


## Sử dụng kfold để tìm và đánh giá dữ liệu
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

class LSTMModel:
    def __init__(self):
        logging.getLogger("Init LSTM Model").setLevel(logging.DEBUG)
        self.data = None
        self.imbalanced_algorithm = None
        self.model = None
        self.start_time = None
        self.end_time = None
        self.kfold = None
        self.perform_training = None

    def set_k_fold(self, k_fold): 
        self.kfold = k_fold
        return self

    def set_data(self, data):
        logging.getLogger("loading").setLevel(logging.INFO)
        self.data = data
        return self
    
    def set_model(self, model): 
        self.model = model
        return self

    def set_imbalance_algorithm(self, imbalance_algorithm):
        self.imbalanced_algorithm = imbalance_algorithm
        return self
    
    def set_perform_training(self, perform_training): 
        self.perform_training = perform_training
        return self
    
    def get_data(self):
        return self.data
    
    def get_start_time(self): 
        return self.start_time
    
    def get_end_time(self):
        return self.end_time
    
    def get_duration_time(self):
        return self.duration_time
    
    def run(self): 
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if(self.data is None):
            raise ValueError("Cung cấp dataset")

        if(self.kfold is None): 
            raise ValueError("Cung cấp thêm thuật toán cross validation")

        if(self.imbalanced_algorithm is None): 
           raise ValueError("Cung cấp thuật toán cân bằng dữ liệu để chạy")

        if(self.model is None): 
            raise ValueError("Cung cấp model để chạy")
        
        if(self.perform_training is None): 
            raise ValueError("Cung cấp luồng để chạy")

        labels = [
            [CONST_ORDINARY_LABEL_NAME],
            CONST_DDOS_LABEL,
            CONST_DOS_LABEL,
            CONST_MIRAI_LABEL,
            CONST_SPOOFING_LABEL,
            CONST_RECON_LABEL,
            CONST_WEB_LABEL,
            CONST_BRUTE_FORCE_LABEL
        ]

        [
            dataset_normal,
            dataset_ddos,
            dataset_dos,
            dataset_mirai,
            dataset_spoofing,
            dataset_recon,
            dataset_web,
            dataset_bruce_force
        ] = split_group_label(data, labels)

        dataset_attack = pd.concat([
            dataset_ddos,
            dataset_dos,
            dataset_mirai,
            dataset_spoofing,
            dataset_recon,
            dataset_web,
            dataset_bruce_force])

        dataset_attack = dataset_attack.drop(['label'], axis = 1)
        dataset_normal = dataset_normal.drop(['label'], axis = 1)

        dataset_attack['label'] = 1
        dataset_normal['label'] = 0

    
        data_two_classes = pd.concat([dataset_normal, dataset_attack])
        print(data_two_classes)

        X = data_two_classes.drop(['label'], axis = True)
        Y = data_two_classes['label']

        ## scaler dữ liệu
        scale_fields = [
            'flow_duration',
            'Header_Length',
            'ack_count',
            'syn_count',
            'fin_count',
            'urg_count',
            'rst_count',
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
            'Weight']

        scaler = MinMaxScaler()
        scaler.fit(X[scale_fields])
        X[scale_fields] = scaler.transform(X[scale_fields])

        self.perform_training(X, Y, self.imbalanced_algorithm)
        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


nm = NearMiss()
model = LSTMModel()
model.set_data(data).set_model(get_LSTM_model).set_k_fold(k_fold=kfold).set_imbalance_algorithm(nm).set_perform_training(perform_LSTM_model).run()

start_time = datetime.strptime(model.get_start_time(), '%Y-%m-%d %H:%M:%S')
end_time = datetime.strptime(model.get_end_time(), '%Y-%m-%d %H:%M:%S')
print(end_time - start_time)
print(model.get_start_time())
print(model.get_end_time())

print(str(model.get_start_time()) + " " + str(model.get_end_time()) + " " + str(model.get_duration_time()))


### Lượng tử hóa 
import pathlib
model = tf.keras.models.load_model(CONST_NAME_MODEL)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

lstm_models_dir = pathlib.Path("/content/lstm_model/")
lstm_models_dir.mkdir(exist_ok=True, parents=True)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
lstm_quant_model = converter.convert()
lstm_model_quant_file = lstm_models_dir/"lstm_model.tflite"
lstm_model_quant_file.write_bytes(lstm_quant_model)

interpreter_quant = tf.lite.Interpreter(model_path=str(lstm_model_quant_file))
interpreter_quant.allocate_tensors()


    

        
