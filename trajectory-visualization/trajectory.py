import pandas as pd
import numpy as np
import visdom
import torch

data_train = pd.read_csv('../data/train.csv')
data_test = pd.read_csv('../data/track1/test.csv')

class trajectory:
    def __init__(self, sat_id):
        self.vis = visdom.Visdom()
        self.sat_id = sat_id
        
    def plot_sim(self, stride_train=0, stride_test=0):
        columns = ['x_sim', 'y_sim', 'z_sim']
        sim_train = data_train[data_train['sat_id'] == self.sat_id].loc[:, columns].reset_index()
        sim_test = data_test[data_test['sat_id'] == self.sat_id].loc[:, columns].reset_index()
        assert (len(sim_train) != 0) or (len(sim_test) != 0), 'No Data for the Specified Satellite ID'
        if (stride_train != 0):
            sim_train = sim_train[sim_train['index'] % stride_train == 0]
        if (stride_test != 0):
            sim_test = sim_test[sim_test['index'] % stride_test == 0]
        sim_train.drop('index', axis=1, inplace=True)
        sim_test.drop('index', axis=1, inplace=True)
        print('Train:', len(sim_train), '| Test:', len(sim_test))
        labels = np.array([1] * len(sim_train) + [2] * len(sim_test))
        sim = sim_train.append(sim_test).values
        self.vis.scatter(sim, labels, opts = {'markersize': 4}, win='sim')
    
    def plot_sim_real(self, stride=0):
        sat_data = data_train[data_train['sat_id'] == self.sat_id].reset_index()
        if (stride != 0):
            sat_data = sat_data[sat_data['index'] % stride == 0]
        labels = np.array([1] * len(sat_data) + [2] * len(sat_data))
        print('Size:', len(labels))
        sim = sat_data.loc[:, ['x_sim', 'y_sim', 'z_sim']].values
        real = sat_data.loc[:, ['x', 'y', 'z']].values
        loc = np.concatenate([sim, real], axis=0)
        assert (loc.shape[0] > 0), 'No Data for the Specified Satellite ID'
        self.vis.scatter(loc, labels, opts = {'markersize': 4}, win='sim-real')