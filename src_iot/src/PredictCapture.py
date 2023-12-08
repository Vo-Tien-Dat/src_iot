import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import pandas as pd
from SendEmail import send_email

def PredictCapture(): 

    CONST_MODEL_PATH_DIRECTORY = '../model/'
    MODEL_FILE_NAME = 'train_modeltest_55.hdf5'
    MODEL_FILE_PATH = CONST_MODEL_PATH_DIRECTORY + MODEL_FILE_NAME
    model = tf.keras.models.load_model(MODEL_FILE_PATH)


    CONST_PATH_DIRECTORY = '../csv/'
    REAL_FILE_NAME = 'capture.csv'
    REAL_FILE_PATH = CONST_PATH_DIRECTORY + REAL_FILE_NAME
    print(REAL_FILE_PATH)
    col_names = ['flow_duration', 'header_length', 'protocol type', 'duration', 'rate',
    'srate', 'drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'http', 'https', 'dns', 'telnet',
    'smtp', 'ssh', 'irc', 'tcp', 'udp', 'dhcp', 'arp', 'icmp', 'ipv', 'llc',
    'tot sum', 'min', 'max', 'avg', 'std', 'tot size', 'iat', 'number',
    'magnitue', 'radius', 'covariance', 'variance', 'weight']
    real_df = pd.read_csv(REAL_FILE_PATH)
    real_df.columns = col_names


    feature_types = [
    {
        'feature_type': 'float32',
        'features': ['flow_duration', 'drate', 'tot sum', 'min', 'max', 'avg', 'std', 'tot size', 'iat', 'number', 'magnitue', 'radius', 'covariance', 'variance', 'weight']
    },
    {
        'feature_type': 'float16',
        'features': ['ack_count', 'syn_count']
    },
    {
        'feature_type': 'uint32',
        'features': ['header_length', 'rate', 'srate']
    },
    {
        'feature_type': 'uint16',
        'features': ['fin_count', 'urg_count', 'rst_count']
    },
    {
        'feature_type': 'bool',
        'features': ['syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'http', 'https', 'dns', 'telnet', 'smtp', 'ssh', 'irc', 'tcp', 'udp', 'dhcp', 'arp', 'icmp', 'ipv', 'llc']
    },
    {
        'feature_type': 'uint8',
        'features': ['duration']
    },
    {
        'feature_type': 'object',
        'features': ['protocol type']
    }
    ]

    for item in feature_types:
        feature_type = item['feature_type']
        features = item['features']
        for feature in features:
            real_df[feature] = real_df[feature].astype(feature_type)


    scaler = StandardScaler()
    real_df[real_df.columns] = scaler.fit_transform(real_df)       
    real_X = real_df.drop(columns = ['tcp','ack_count','https','fin_flag_number','syn_flag_number','psh_flag_number','rst_flag_number','fin_count','http','ssh','dns','llc','ipv','arp','drate','ece_flag_number','cwr_flag_number','dhcp','irc','smtp','telnet'])

    scaler2 = MinMaxScaler(feature_range=(0, 1))
    scaler2.fit_transform(real_X)

    real_Y = model.predict(real_X)
    real_label = [tf.argmax(y, axis=0).numpy() for y in real_Y]

    mapping_label = {
        0: 'benign',
        1: 'Bruce Force', 
        2: 'DDoS', 
        3: 'Dos',
        4: 'Mirai', 
        5: 'Recon', 
        6: 'Spoofing',
        7: 'Web'
    }

    df_real = pd.DataFrame(real_label)
    df_real[0] = df_real[0].replace(mapping_label)

    return df_real.value_counts().reset_index().to_numpy()
