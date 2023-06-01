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
    adjacency_matrix = np.zeros((214, 214, 3))  

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
    

    return adjacency_matrix
#%%
Number_gen = 20000 #以k為單位
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

#%%
file_name = f'train_matrix_{Number_gen/1000}k_3f.npz'
adjacency_matrices = np.array(adjacency_matrices)

# 將adjacency_matrices保存為NPZ檔案
np.savez(file_name, adjacency_matrices=adjacency_matrices)

#%%
label = [data_source[i].y_relaxed for i in range(Number_gen)]
label_name = f'matrix_label_{Number_gen/1000}k_3f.txt'

with open(label_name, 'w') as file:
    for item in label:
        file.write(f'{item}\n')


