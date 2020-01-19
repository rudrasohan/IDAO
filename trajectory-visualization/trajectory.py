import pandas as pd
import numpy as np
import visdom

class trajectory:
    def __init__(self, sat_id, data_train, data_test):
        self.vis = visdom.Visdom()
        self.sat_id = sat_id
        self.train_sim = data_train[data_train['sat_id'] == sat_id].loc[:, ['x_sim', 'y_sim', 'z_sim']].reset_index(drop=True)
        self.test_sim = data_test[data_test['sat_id'] == sat_id].loc[:, ['x_sim', 'y_sim', 'z_sim']].reset_index(drop=True)
        self.train_real = data_train[data_train['sat_id'] == sat_id].loc[:, ['x', 'y', 'z']].reset_index(drop=True)
        
    def plot_sim(self, stride_train=1, stride_test=1):
        assert (len(self.train_sim) != 0), 'No Data for the Specified Satellite ID'
        sim_train = self.train_sim[self.train_sim.index % stride_train == 0]
        sim_test = self.test_sim[self.test_sim.index % stride_test == 0]
        sim = sim_train.append(sim_test).values
        labels = np.array([1] * len(sim_train) + [2] * len(sim_test))
        print('Train:', len(sim_train), '| Test:', len(sim_test))
        self.vis.scatter(sim, labels, opts = {'markersize': 4}, win='sim')
    
    def plot_sim_real(self, stride=0):
        assert (len(self.train_sim) != 0), 'No Data for the Specified Satellite ID'
        sim_train = self.train_sim[self.train_sim.index % stride == 0].values
        real_train = self.train_real[self.train_real.index % stride == 0].values
        data = np.concatenate([sim_train, real_train], axis=0)
        labels = np.array([1] * len(sim_train) + [2] * len(real_train))
        print('Sample Size Sim/Real:', len(labels))
        self.vis.scatter(data, labels, opts = {'markersize': 4}, win='sim-real')