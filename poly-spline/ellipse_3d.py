import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def get_ellipse_data(data, sat_id, ellipse_id, sim):
    cols = ['x', 'y', 'z']
    if (sim):
        cols = [(axis + '_sim') for axis in cols]
    ellipse_data = data[data['sat_id'] == sat_id].loc[:, cols]
    ellipse_data = ellipse_data.iloc[ellipse_id * 24 : (ellipse_id + 1) * 24, :].values
    ellipse_mean = ellipse_data.mean(axis=0)
    ellipse_data -= ellipse_mean
    return ellipse_data

def get_rotation_parameters(ellipse_data):
    ellipse_normal = np.cross(ellipse_data[0]-ellipse_data[1], ellipse_data[0]-ellipse_data[2])
    ellipse_normal /= np.linalg.norm(ellipse_normal)
    if (ellipse_normal[2] < 0):
        ellipse_normal *= -1
    z_axis = np.array([0,0,1])
    phi = np.arccos(np.dot(ellipse_normal, z_axis))
    rotation_axis = np.cross(ellipse_normal, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    return [ellipse_normal, rotation_axis, phi]

def rodrigues_rotate(vect, axis, phi):
    z_axis = np.array([0,0,1])
    vect_rotated = vect * np.cos(phi)
    vect_rotated += np.cross(axis, vect) * np.sin(phi)
    vect_rotated += axis * np.dot(z_axis, vect) * (1 - np.cos(phi))
    return vect_rotated

def transform3d(data, sat_id, ellipse_id, sim=True):
    ellipse_data = get_ellipse_data(data, sat_id, ellipse_id, sim)
    ellipse_normal, rotation_axis, phi = get_rotation_parameters(ellipse_data)
    transformed_data = np.array([rodrigues_rotate(vect, rotation_axis, phi) for vect in ellipse_data])
    transformed_data = transformed_data[:,[0,1]]
    return transformed_data

class Ellipse(nn.Module):
    
    def __init__(self, name, major, minor, cx, cy, theta):
        super(Ellipse, self).__init__()
        self.name = name
        self.a = nn.Parameter(1/(major**2))
        self.b = nn.Parameter(1/(minor**2))
        self.cx = nn.Parameter(cx)
        self.cy = nn.Parameter(cy)
        self.theta = nn.Parameter(theta)
    
    def forward(self, data):
        x_rot = data[:,0] - self.cx
        y_rot = data[:,1] - self.cy
        x = x_rot * torch.cos(self.theta) - y_rot * torch.sin(self.theta)
        y = x_rot * torch.sin(self.theta) + y_rot * torch.cos(self.theta)
        res = (self.a * x**2 + self.b * y**2 - 1) ** 2
        return [res, x, y]
    
    def plot(self, data):
        major = np.sqrt(1 / self.a.detach().numpy())
        minor = np.sqrt(1 / self.b.detach().numpy())
        theta = -self.theta.detach().numpy()
        cx = self.cx.detach().numpy()
        cy = self.cy.detach().numpy()
        print('Major: {} | Minor: {} | Center: ({},{}) | Rotation: {}'.format(major, minor, cx, cy, theta*180/np.pi))
        generated = []
        for x in np.arange(-major, major, 0.01):
            y = np.sqrt((1 - x**2 / major**2) * minor**2)
            generated.append([x,y])
            generated.append([x,-y])
        generated = np.array(generated)
        generated_transformed = np.transpose(np.array([generated[:,0] * np.cos(theta) - generated[:,1] * np.sin(theta),
                                                       generated[:,0] * np.sin(theta) + generated[:,1] * np.cos(theta)]))
        generated_transformed += [cx, cy]
        plt.scatter(data[:,0], data[:,1])
        plt.scatter(generated_transformed[:,0], generated_transformed[:,1], c='green', s=0.5)
        return

def get_initial_values(data):
    euclidian = []
    pairs = []
    for i in range(23):
        for j in range(i + 1, 24):
            euclidian_distance = np.linalg.norm(data[i] - data[j])
            pairs.append([i, j])
            euclidian.append(euclidian_distance)
    max_index = np.argmax(euclidian)
    major_axis = data[pairs[max_index][0]] - data[pairs[max_index][1]]
    if (major_axis[1] < 0):
        major_axis *= -1
    major = np.linalg.norm(major_axis) / 2
    center = (data[pairs[max_index][0]] + data[pairs[max_index][1]]) * 0.5
    theta = np.arccos(np.dot(major_axis, [1,0]) / (2 * major))
    return [major, major, center[0], center[1], -theta]    

def fit_ellipse(transformed_data, iterations=1500, learning_rate=0.001, name=''):
    major, minor, cx, cy, theta = [torch.tensor(val).float() for val in get_initial_values(transformed_data)]
    model = Ellipse(name, major, major, cx, cy, theta)
    optim = torch.optim.Adam(model.parameters(), learning_rate)
    data = torch.from_numpy(transformed_data).float()
    for itr in range(2000):
        optim.zero_grad()
        pred, _, _ = model(data)
        loss = torch.sum(pred)
        loss.backward()
        optim.step()
    print('Final Loss: ', loss.item())
    return model