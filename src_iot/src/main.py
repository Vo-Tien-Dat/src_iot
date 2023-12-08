import threading
import pandas as pd
from WatchFolder import WatchFolder
from log.Generating_dataset import GenerateDataset
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from PredictCapture import PredictCapture
from SendEmail import send_email


def thread_folderCapture(src_path, callback):
    WatchFolder(src_path=src_path, callback=callback)

def callbackConvertCSV(src_path):
    print('Phan tich du lieu file csv')
    table_data = PredictCapture()
    send_email(table_data=table_data)


def callbackCapturePcap(src_path):
    print(src_path)
    GenerateDataset(['../capture/capture.pcap'])
    
if __name__ == "__main__":
    print("Khởi tạo chương trình")
    abs_path = os.getcwd()
  
    capture_folders = [
        {
            'folder': '../captures', 
            'callback': callbackCapturePcap
        }, 
        {
            'folder': '../csv', 
            'callback': callbackConvertCSV
        }
    ]

    threads = []

    for capture_folder in capture_folders:
        folder = capture_folder['folder']
        callback = capture_folder['callback']
        threadFolderCapture = threading.Thread(target=thread_folderCapture, args=(folder,callback))
        threadFolderCapture.start()
        threads.append(threadFolderCapture)

    for thread in threads:
        thread.join()

    
