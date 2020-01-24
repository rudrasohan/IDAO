import pandas as pd
import numpy as np
from datetime import datetime

def format_time(data):
    time = []
    for t in data['epoch']:
        time.append(datetime.strptime(t,"%Y-%m-%dT%H:%M:%S.%f"))
    time = np.array(time)
    time -= datetime(2014, 1, 1)
    time = [t.total_seconds() for t in time]
    return time

def scale_data(data_train, data_test, scale):
    cols_sim = ['x_sim', 'y_sim', 'z_sim', 'epoch']
    cols_real = ['x', 'y', 'z']
    for col in cols_sim:
        data_train[col] /= scale
        data_test[col] /= scale
    for col in cols_real:
        data_train[col] /= scale
        
def generate_labels(n):
    labels = []
    for i in range(0,n):
        labels.append(i%24 + 1)
    label = np.array(labels)
    return labels

def cluster_data(data_train, data_test):
    clusters_train = np.array([])
    clusters_test = np.array([])
    for sat_id in range(600):
        sat_train = data_train[data_train['sat_id'] == sat_id]
        sat_test = data_test[data_test['sat_id'] == sat_id]
        sep = sat_train.shape[0]
        labels = generate_labels(sep + sat_test.shape[0])
        clusters_train = np.concatenate([clusters_train, labels[:sep]])
        clusters_test = np.concatenate([clusters_test, labels[sep:]])
    data_train['cluster'] = clusters_train
    data_test['cluster'] = clusters_test
    
def process_sat_data(data_train, data_test, scale=10000):
    data_train['epoch'] = format_time(data_train)
    data_test['epoch'] = format_time(data_test)
    scale_data(data_train, data_test, scale)
    cluster_data(data_train, data_test)