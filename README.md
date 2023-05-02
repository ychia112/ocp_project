# Open Catalyst 2022 project
The dataset of this project (OC2022 IS2RE) is obtained from here : [Open Catalyst Project](https://opencatalystproject.org/)

## Project Description

This project aims to predict the relaxed energy of a material's structure using various machine learning techniques. The goal is to eliminate the need for tedious DFT calculations and iterations, thereby increasing the efficiency of designing electrocatalysts.

## Process Details
### I. Setting Environment
Follow the link to set environment : [Installation](https://github.com/Open-Catalyst-Project/ocp/blob/main/INSTALL.md)  
I had some errors when setting up the environment and the solutions are down below. Hope it helps!

**Step1.** There might be error in the step ```mamba env create -f env.yml```, then you can manually install
    all the dependencies in the env.yml file.

**Step2.** Use ```conda list``` to find the version of "torch" and "python", then go to the website:
    https://pytorch-geometric.com/whl/
    to find the "torch sparse" and "torch scatter" file, "cpxx" means the python version.

**Step3.** Install "torch sparse" and "torch scatter" file and use ```pip install "C:Path to the file"``` to
    install those packages.
    
Done!!

### II. Download dataset from OC22
Follow the code and tutorials form Open Catalyst github : https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md
The dataset needed in this project is Open Catalyst 2022(OC22) "Initial Structure to Relaxed Structure (IS2RS) and Initial Structure to Relaxed Total Energy (IS2RE-Total) tasks"

After Downloaded the needed files, you can start using the code provided in this repository.

### III. System Building (buildSystem.py)
The ML method I'm using in this project is 3D Convolution Neural Network, so in this step we have to build the 3D grid world to store the informations, then use these systems to train/test the ML model.

This is the composition of OC22 IS2RE dataset:
* `data/oc22/is2re-total/train/:          45,890` 
* `data/oc22/is2re-total/val_id/:          2,624` 
* `data/oc22/is2re-total/val_ood/:         2,780` 
* `data/oc22/is2re-total/test_id/:         2,624` 
* `data/oc22/is2re-total/test_ood/:        2,650` 

To reduce the computation cost, I only use the data in `/train`.

After we read the LMDB file, we can get info of those data from ```data[objectID]```. 
Each Data object includes the following information for each corresponding system (assuming K atoms):

* `sid` - [1] System ID corresponding to each structure
* `atomic_numbers` - [K x 1] Atomic numbers of all atoms in the system
* `pos` - [K x 3] Initial structure positional information of all atoms in the system (x, y, z cartesian coordinates)
* `natoms` - [1] Total number atoms in the system
* `cell` -  [3  x 3] System unit cell (necessary for periodic boundary condition (PBC) calculations)
* `tags` - [K x 1] Atomic tag information: 0 - Sub-surface atoms, 1 - Surface atoms 2 - Adsorbate atoms
* `fixed` - [K x 1]  Whether atoms were kept fixed or free during DFT. 0 - free, 1 - fixed
* `nads` - [1] Total number of adsorbates in the system
* `oc22` - [1] Whether system is from the OC22 dataset, 1 - OC22, 0 - Not OC22

Train/Val LMDBs additionally contain the following attributes:

* `y_relaxed` - [1] DFT energy of the relaxed structure
* `pos_relaxed` - [K x 3] Relaxed structure positional information of all atoms in the system (x, y, z cartesian coordinates)

**Build 3D grid world**  
This code is designed to assign atomic properties to a 3D grid based on the distance of each grid point to the nearest atom. The algorithm works by calculating the distance between each grid point and all the atoms in the system and selecting the shortest distance. If this distance is less than the radius of the atom, the grid point is assigned the properties of that atom, including its atomic weight.

**Save the data**  
The training data is saved as `'system_models.npz'`, and the label (y_relaxed, which is the relaxed energy) is saved as a `.py` file, it is convenient to import it in the 3Dconv.py code.

> **Note**  
> There are 500 3D grid systems are built, feel free to contact me and get the data for testing!

### IV. 3D Convolution Neural Network (3Dconv.py)

After building the 3D grid system, we can start using the data to do 3D convolution.
I've done this by using the package in Tensorflow.



