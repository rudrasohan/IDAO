
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:

import torch
import torch.nn as nn
import visdom
import os


# In[ ]:

from sklearn.decomposition import PCA


# In[ ]:

from preprocessing_utils import process_sat_data


# In[ ]:

data_train = pd.read_csv('../data/train.csv')
data_test = pd.read_csv('../data/track1/test.csv')


# In[ ]:

train_data = data_train.copy(deep=True)
test_data = data_test.copy(deep=True)


# In[ ]:

process_sat_data(train_data, test_data, scale=10000)


# # Arrange Data: 24-Point Ellipse Format

# In[ ]:

def get_sat_ellipse_ids(n):
    ids = []
    incomplete = (n%24)
    for i in range(n-incomplete):
        ids.append(int(i/24))
    prev = ids[-1]
    for i in range(incomplete):
        ids.append(prev)
    return ids


# In[ ]:

def generate_ellipse_ids(data):
    ids = []
    counts = np.unique(data['sat_id'], return_counts=True)[1]
    for count in counts:
        sat_ellipse_ids = get_sat_ellipse_ids(count)
        ids += sat_ellipse_ids
    return ids


# In[ ]:

def arrange_data(train_data, test_data):
    cols_sim = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim', 'sat_id', 'id']
    cols_real = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz', 'sat_id']
    data_sim = train_data.loc[:, cols_sim].append(test_data.loc[:, cols_sim])
    data_sim = data_sim.sort_values('id').drop('id', axis=1)
    data_real = train_data.loc[:, cols_real]
    data_sim['ellipse_id'] = generate_ellipse_ids(data_sim)
    data_real['ellipse_id'] = generate_ellipse_ids(data_real)
    return [data_sim, data_real]


# In[ ]:

data_sim, data_real = arrange_data(train_data, test_data)


# In[ ]:

def format_data(pos, vel):
    last_index = int(len(pos)/24) * 24
    pos = pos[:last_index, :].reshape(-1,24,3)
    vel = vel[:last_index, :].reshape(-1,24,3)
    return [pos, vel]


# In[ ]:

pos_sim = {}
vel_sim = {}
pos_real = {}
vel_real = {}


# In[ ]:

for sat_id in range(600):
    sat_sim = data_sim[data_sim['sat_id'] == sat_id].loc[:, ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].values
    sat_real = data_real[data_real['sat_id'] == sat_id].loc[:, ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']].values
    pos_sim[sat_id], vel_sim[sat_id] = format_data(sat_sim[:, [0,1,2]], sat_sim[:, [3,4,5]])
    pos_real[sat_id], vel_real[sat_id] = format_data(sat_real[:, [0,1,2]], sat_real[:, [3,4,5]])


# # Compute Transformation

# 3D_Point . Inv(M) = 2D_Point <br>
# 2D_Point . M = 3D_Point

# In[ ]:

class Ellipse(nn.Module):
    
    def __init__(self, name, a, b, cx, cy):
        super(Ellipse, self).__init__()
        self.name = name
        self.alpha = nn.Parameter(1/(a**2))
        self.beta = nn.Parameter(1/(b**2))
        self.cx = nn.Parameter(cx)
        self.cy = nn.Parameter(cy)
        self.loss = -1
    
    def forward(self, data):
        x = data[:, 0] - self.cx
        y = data[:, 1] - self.cy
        res = (self.alpha * x**2 + self.beta * y**2 - 1) ** 2
        return res
    
    def get_parameters(self):
        a = np.sqrt(1/self.alpha.item())
        b = np.sqrt(1/self.beta.item())
        cx = self.cx.item()
        cy = self.cy.item()
        loss = self.loss
        return [a, b, loss]


# In[ ]:

def fit_ellipse(sat_id, ellipse_id, transformed, iterations=2500, learning_rate=0.001):
    transformed = torch.from_numpy(transformed).float()
    a = 0.5 * (transformed[:, 0].max() - transformed[:, 0].min())
    b = 0.5 * (transformed[:, 1].max() - transformed[:, 1].min())
    cx = torch.mean(transformed[:, 0])
    cy = torch.mean(transformed[:, 1])
    name = 's' + str(sat_id) + 'e' + str(ellipse_id)
    ellipse = Ellipse(name, a, b, cx, cy)
    optim = torch.optim.Adam(ellipse.parameters(), learning_rate)
    for itr in range(iterations):
        optim.zero_grad()
        pred = ellipse(transformed)
        loss = torch.sum(pred)
        loss.backward()
        optim.step()
    ellipse.loss = loss.item()
    return ellipse


# In[ ]:

def get_pca_estimates(ellipse_data):
    pca = PCA(3)
    pca.fit(ellipse_data)
    transformation = np.append(pca.components_, [ellipse_data.mean(axis=0)], axis=0)
    transformation = np.concatenate([transformation, [[0],[0],[0],[1]]], axis=1)
    return transformation


# In[ ]:

def compute_transformation(sat_id, ellipse_id, ellipse_data):
    transformation = get_pca_estimates(ellipse_data)
    ellipse_data = np.concatenate([ellipse_data, np.ones((24,1))], axis=1)
    transformed_data = ellipse_data.dot(np.linalg.inv(transformation))
    ellipse = fit_ellipse(sat_id, ellipse_id, transformed_data, 2500, 0.001)
    center_2d = np.array([ellipse.cx, ellipse.cy, 0, 1])
    center_3d = center_2d.dot(transformation)
    transformation[-1] = center_3d
    return [transformation, ellipse]


# # Data Generation

# Format: <br>
# sat_id, ellipse_id, major, minor, loss, rotation (3x3), translation (1x3)

# In[ ]:

path = '../data/'


# In[ ]:

def get_checkpoint(path, file):
    file_list = os.listdir(path)
    if file not in file_list:
        return [0,0]
    df = pd.read_csv(path + file)
    last_sat_id, last_ellipse_id = df.iloc[-1,[0,1]].values.astype(int)
    return [last_sat_id, last_ellipse_id]


# In[ ]:

def generate_data(data, path, file):
    last_sat_id, last_ellipse_id = get_checkpoint(path, file)
    if (last_sat_id == 0 and last_ellipse_id == 0):
        generated_data = []
    else:
        generated_data = list(pd.read_csv(path + file).values)
    for sat_id, sat_data in data.items():
        if (sat_id < last_sat_id):
            continue
        for ellipse_id, ellipse_data in enumerate(sat_data):
            if (ellipse_id <= last_ellipse_id):
                continue
            else:
                last_ellipse_id = -1
            transformation, ellipse = compute_transformation(sat_id, ellipse_id, ellipse_data)
            R = transformation[:3, :3].ravel()
            T = transformation[3, :3]
            ids = [sat_id, ellipse_id]
            ellipse_parameters = ellipse.get_parameters()
            row = np.concatenate([ids, ellipse_parameters, R, T], axis=0)
            generated_data.append(row)
            print('Satellite ID: {}, Ellipse Id: {} | Loss: {} [Sim]'.format(sat_id, ellipse_id, ellipse_parameters[-1]))
            df = pd.DataFrame(generated_data)
            df.to_csv(path + file, index=False)
        print ('Satellite ID', sat_id, 'Saved')


# In[ ]:

#generate_data(pos_sim, path, 'data_sim.csv')


# In[ ]:

generate_data(pos_real, path, 'data_real.csv')

