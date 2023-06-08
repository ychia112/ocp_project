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
#%%
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

def create_adjacency_matrix(atom_coordinates, center, atomic_number):
    num_atoms = len(atom_coordinates)
    adjacency_matrix = np.zeros((num_atoms, num_atoms, 2))  
    
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
    
    original_size = num_atoms
    original_matrix = adjacency_matrix

    new_size = 216
    new_matrix = np.zeros((new_size, new_size, 2))

    scale = (new_size - 1) / np.log(original_size)

    for i in range(new_size):
        for j in range(new_size):

            original_x = int(np.exp(i / scale))
            original_y = int(np.exp(j / scale))
            

            original_x = np.clip(original_x, 0, original_size - 1)
            original_y = np.clip(original_y, 0, original_size - 1)
            
            if original_matrix[original_x, original_y,0] == 0:
                new_matrix[i, j, 0] = 0
            if original_matrix[original_x, original_y,1] == 0:
                new_matrix[i, j, 1] = 0
            else:
                new_matrix[i, j, 0] = np.log(original_matrix[original_x, original_y,0])
                new_matrix[i, j, 1] = np.log(original_matrix[original_x, original_y,1])

    return adjacency_matrix, new_matrix

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
    adjacency_matrix, new_matrix = create_adjacency_matrix(atom_coordinates, center, atomic_number)
    adjacency_matrices.append(new_matrix)

adjacency_matrices = np.array(adjacency_matrices)

# check the shape
print(adjacency_matrices.shape)

#%% save the data
file_name = f'train_matrix_{int(Number_gen/1000)}k_2f_log.npz'
np.savez(file_name, adjacency_matrices=adjacency_matrices)

#%% save the label
label = [data_source[i].y_relaxed for i in range(Number_gen)]
label_name = f'matrix_label_{int(Number_gen/1000)}k_2f_log.txt'

with open(label_name, 'w') as file:
    for item in label:
        file.write(f'{item}\n')

#%% visualize
import matplotlib.pyplot as plt
ad_mat, nw_mat = create_adjacency_matrix(atom_coordinates, center, atomic_number)
plt.subplot(1, 2, 1)
plt.imshow(ad_mat[:,:,0], cmap='viridis')
plt.title('Original Matrix')

plt.subplot(1, 2, 2)
plt.imshow(nw_mat[:,:,0], cmap='viridis')
plt.title('Transformed Matrix')

