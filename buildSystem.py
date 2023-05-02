# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:07:00 2023

@author: Yu Chia Cheng
"""

import string
import os
import numpy as np
from ocpmodels.datasets import LmdbDataset
import matplotlib.pyplot as plt
import random
from ase import Atoms
from scipy.spatial.distance import cdist
from mendeleev import element
import numba
from numba import cuda, jit, njit
import time

#%%  The OC2022 dataset for IS2RE

test_id = LmdbDataset({"src": "data/oc22/is2re-total/test_id"})
test_ood = LmdbDataset({"src": "data/oc22/is2re-total/test_ood"})
train = LmdbDataset({"src": "data/oc22/is2re-total/train"})
val_id = LmdbDataset({"src": "data/oc22/is2re-total/val_id"})
val_ood = LmdbDataset({"src": "data/oc22/is2re-total/val_ood"})
# data_source data
data_source = train
#%%=============================FOR TEST DATA==================================
#%%
def findGridSize(case): #This is for finding the proper system grid size
    cells = data_source[case].cell[0].tolist()
    spacing = 0.5   # angstroms # set the spacing here
    nx, ny, nz = np.ceil(np.diag(cells) / spacing).astype(int)
    return (nx, ny, nz)

result = []
maxx = 0  
maxy = 0  
maxz = 0
for i in range(len(data_source)):
    x, y, z = findGridSize(i)
    result.append((x, y, z))
    if x > maxx:
        maxx = x
    if y > maxy:
        maxy = y
    if z > maxz:
        maxz = z
        
print(f'The size of the grid world can be set as ({maxx}, {maxy}, {maxz})')
del result, x, y, z, i
#%%
@jit #@jit is for accelerate the computation speed using Numba, some warnings might pop up, just ignore them.
def getsymbols(case,atomlist): 
    symbols = [element(int(atomic_number)).symbol for atomic_number in atomlist]
    return symbols
@jit
def systemGridWorld(case: int, data_source: LmdbDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # grid world building
    positions = data_source[case].pos.tolist()
    cells = data_source[case].cell[0].tolist()
    atomic_num = data_source[case].atomic_numbers.tolist()

    #nx, ny, nz = np.ceil(np.diag(cells) / spacing).astype(int)
    nx, ny, nz = maxx, maxy, maxz
    # Create a meshgrid for the x, y, and z coordinates
    x, y, z = [np.linspace(0, L, N) for L, N in zip(np.diag(cells), (nx, ny, nz))]
    X, Y, Z = np.meshgrid(x, y, z)

    # Stack the meshgrid coordinates into a single (N, 3) array
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Calculate the distance from each grid point to every atom
    distances = cdist(grid_points, positions)

    atomic_radii = [round(element(int(symbol)).atomic_radius_rahm, 2) for symbol in atomic_num]
    for distance in distances:
        for i, n in enumerate(distance):
            distance[i] = n - atomic_radii[i]/1000
            

    # Find the index of the closest atom to each grid point
    closest_atom_index = np.argmin(distances, axis=1)
    closest_atom_distance = np.min(distances, axis = 1)

    # Modify closest_atom_index based on closest_atom_distance
    for i in range(len(closest_atom_index)):
        if closest_atom_distance[i] > 0:
            closest_atom_index[i] = -1


    # Assign the atomic symbol of the closest atom to each grid point
    grid_labels = np.where(closest_atom_index == -1, 0, np.array(atomic_num)[closest_atom_index]) 

    # Reshape the grid labels into the same shape as the meshgrid
    grid_labels = grid_labels.reshape((nx, ny, nz))
    
    return closest_atom_index, grid_labels, grid_points

#%% 平均的取sample (by y_relaxed)

import numpy as np

y_relaxed = [d.y_relaxed for d in data_source]
#%%

num_bins = 25
# Divide the data into equal-size bins
hist, bins = np.histogram(y_relaxed, bins=num_bins)
bin_indices = np.digitize(np.array(y_relaxed), bins)
num_samples_per_bin = 30  # choose the number of samples to take from each bin
sampled_indices = []
for i in range(len(bins) - 1):
    bin_data_indices = np.where(bin_indices == i)[0]
    bin_size = len(bin_data_indices)
    if bin_size == 0:
        continue
    sample_size = min(bin_size, num_samples_per_bin)
    sampled_bin_data_indices = np.random.choice(bin_data_indices, size=sample_size, replace=False)
    sampled_indices.append(sampled_bin_data_indices)
sampled_indices = np.concatenate(sampled_indices)
y_relaxed_sample = []
for i in sampled_indices:
    y_relaxed_sample.append(y_relaxed[i])
    
#%%

import warnings
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)
start_time = time.time()

system_models = []
for case in sampled_indices:
    atomlist = data_source[case].atomic_numbers.tolist()
    closest_atom_index, system_model, grid_points = systemGridWorld(case,data_source)
    system_model = system_model.astype(int)
    system_models.append(system_model)

np.savez('system_models.npz', *system_models)
end_time = time.time()

total_time = end_time - start_time
print(f"Execution time: {total_time} seconds")
#%%
"""
#This is for testing the computing time for building 1 grid system, you can skip this block.
import warnings
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)

start_time = time.time()
case = 0
closest_atom_index, system_model, grid_points = systemGridWorld(case, data_source)
system_model = system_model.astype(int)

end_time = time.time()
total_time = end_time - start_time
print(f"Execution time: {total_time} seconds")
"""
#%% get the labels of data in system_model.npz
system_labels = []
for case in sampled_indices:
    system_labels.append(data_source[case].y_relaxed)
#after running this code above, you have to create a .py file and store this variable.

#%% plot the relaxed energy distribution
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.hist(y_relaxed, bins=25)
plt.xlim(-2000, 500)
plt.ylim(0, 8500)
plt.title('Dataset y_relaxed histogram')
plt.xlabel('Relaxed Energy')
plt.ylabel('counts')

fig = plt.figure(figsize=(8, 6), dpi=300)
plt.hist(y_relaxed_sample)
plt.xlim(-2000, 500)
plt.ylim(0, 8500)
plt.title('Sampled y_relaxed histogram')
plt.xlabel('Relaxed Energy')
plt.ylabel('counts')

fig = plt.figure(figsize=(8, 6), dpi=300)
plt.hist(y_relaxed_sample)
plt.xlim(-2000, 500)
plt.title('Sampled y_relaxed histogram')
plt.xlabel('Relaxed Energy')
plt.ylabel('counts')
