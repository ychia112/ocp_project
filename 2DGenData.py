# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:44:27 2023

@author: yu chia cheng
"""


import numpy as np
from ocpmodels.datasets import LmdbDataset
import math

test_id = LmdbDataset({"src": "data/oc22/is2re-total/test_id"})
test_ood = LmdbDataset({"src": "data/oc22/is2re-total/test_ood"})
train = LmdbDataset({"src": "data/oc22/is2re-total/train"})
val_id = LmdbDataset({"src": "data/oc22/is2re-total/val_id"})
val_ood = LmdbDataset({"src": "data/oc22/is2re-total/val_ood"})
# data_source data
data_source = train
#%% functions of generating features

def calculate_bond_length(atom1, atom2):
    x1, y1, z1 = atom1
    x2, y2, z2 = atom2
    bond_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return bond_length

def calculate_bond_angle(atom1, atom2, atom3):
    x1, y1, z1 = atom1
    x2, y2, z2 = atom2
    x3, y3, z3 = atom3
    v1 = [x2 - x1, y2 - y1, z2 - z1]
    v2 = [x2 - x3, y2 - y3, z2 - z3]
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a**2 for a in v1))
    magnitude_v2 = math.sqrt(sum(a**2 for a in v2))
    try :
        bond_angle = math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))
    except ValueError:
        bond_angle = 0
    return bond_angle

#%% generate fft features (這部分是產生目前第4,5,6個feature的function)
import matplotlib.pyplot as plt
data = data_source[1].pos.tolist()

def gen_fft(size, data):
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    z = [point[2] for point in data]
    # Create grid for xy plane
    xy_grid_size = size
    xy_x_grid, xy_y_grid = np.meshgrid(np.linspace(min(x), max(x), xy_grid_size),
                                   np.linspace(min(y), max(y), xy_grid_size))
    xy_z_grid = np.zeros_like(xy_x_grid)
    
    for point in data:
        xy_x, xy_y, xy_z = point
        x_idx = int((xy_x - min(x)) / (max(x) - min(x)) * (xy_grid_size - 1))
        y_idx = int((xy_y - min(y)) / (max(y) - min(y)) * (xy_grid_size - 1))   
        xy_z_grid[y_idx, x_idx] = xy_z

    # Perform FFT on xy plane
    xy_fft = np.fft.fft2(xy_z_grid)
    xy_fft = np.fft.fftshift(xy_fft)

    # Create grid for xz plane
    xz_grid_size = size
    xz_x_grid, xz_z_grid = np.meshgrid(np.linspace(min(x), max(x), xz_grid_size),
                                   np.linspace(min(z), max(z), xz_grid_size))
    xz_y_grid = np.zeros_like(xz_x_grid)

    for point in data:
        xz_x, xz_y, xz_z = point
        x_idx = int((xz_x - min(x)) / (max(x) - min(x)) * (xz_grid_size - 1))
        z_idx = int((xz_z - min(z)) / (max(z) - min(z)) * (xz_grid_size - 1))
        xz_y_grid[z_idx, x_idx] = xz_y

    # Perform FFT on xz plane
    xz_fft = np.fft.fft2(xz_y_grid)
    xz_fft = np.fft.fftshift(xz_fft)


    # Create grid for yz plane
    yz_grid_size = size
    yz_y_grid, yz_z_grid = np.meshgrid(np.linspace(min(y), max(y), yz_grid_size),
                                   np.linspace(min(z), max(z), yz_grid_size))
    yz_x_grid = np.zeros_like(yz_y_grid)

    for point in data:
        yz_x, yz_y, yz_z = point
        y_idx = int((yz_y - min(y)) / (max(y) - min(y)) * (yz_grid_size - 1))
        z_idx = int((yz_z - min(z)) / (max(z) - min(z)) * (yz_grid_size - 1))
        yz_x_grid[z_idx, y_idx] = yz_x

    # Perform FFT on yz plane
    yz_fft = np.fft.fft2(yz_x_grid)
    yz_fft = np.fft.fftshift(yz_fft)
    return xy_fft, xz_fft, yz_fft, xy_z_grid, xz_y_grid, yz_x_grid




#%%

def create_adjacency_matrix(atom_coordinates, center, atomic_number):
    num_atoms = len(atom_coordinates)
    adjacency_matrix = np.zeros((214, 214, 6))   # 214*214代表2D CNN訓練數據大小, 6代表目前有6個features

    for i in range(num_atoms):
        
        for j in range(i+1, num_atoms):
            atom_i = atom_coordinates[i]
            atom_j = atom_coordinates[j]
            distance = calculate_bond_length(atom_i, atom_j)
            bond_angle = calculate_bond_angle(atom_i, atom_j, center) 
            adjacency_matrix[i, j, 0] = distance
            adjacency_matrix[i, j, 1] = bond_angle
            adjacency_matrix[j, i, 0] = distance
            adjacency_matrix[j, i, 1] = bond_angle
            
    for i in range(num_atoms):
        adjacency_matrix[i, i, 2] = atomic_number[i]
        adjacency_matrix[:, :, 4], adjacency_matrix[:, :, 5], adjacency_matrix[:, :, 6],_,_,_ = gen_fft(214, data_source[i].pos.tolist())
            

    return adjacency_matrix
#%%
Number_gen = 10000 #以k為單位
adjacency_matrices = []

for case in range(Number_gen):
    atomic_number = data_source[case].atomic_numbers.tolist()
    atom_coordinates = data_source[case].pos.tolist()
    x_values, y_values, z_values = zip(*atom_coordinates)

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    z_mean = sum(z_values) / len(z_values)
    center = (x_mean, y_mean, z_mean)
    adjacency_matrix = create_adjacency_matrix(atom_coordinates, center, atomic_number)
    adjacency_matrices.append(adjacency_matrix)

adjacency_matrices = np.array(adjacency_matrices)

# 檢查形狀
print(adjacency_matrices.shape)

#%% save data (you can set the file name you want)
file_name = f'train_matrix_{int(Number_gen/1000)}k_3f.npz' #xf means x features, xk means xk training data
adjacency_matrices = np.array(adjacency_matrices)
np.savez(file_name, adjacency_matrices=adjacency_matrices)

#%%
label = [data_source[i].y_relaxed for i in range(Number_gen)]
label_name = f'matrix_label_{int(Number_gen/1000)}k_3f.txt' 

with open(label_name, 'w') as file:
    for item in label:
        file.write(f'{item}\n')

#%% useful code (visualize):
#%% coordinate visualize
plt.figure()
plt.scatter(x,y)
plt.title("XY Plane Projection")

plt.figure()
plt.scatter(y,z)
plt.title("YZ Plane Projection")

plt.figure()
plt.scatter(x,z)
plt.title("XZ Plane Projection")
#%% training data visualaize
# Plotting xy plane
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.abs(xy_fft))
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_title('FFT XY Projection')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.abs(xz_fft))
ax.set_xlabel('kx')
ax.set_ylabel('kz')
ax.set_title('FFT XZ Projection')
# Plotting yz plane
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.abs(yz_fft))
ax.set_xlabel('ky')
ax.set_ylabel('kz')
ax.set_title('FFT YZ Projection')

plt.show()

