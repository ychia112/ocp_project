# Open Catalyst 2022 project
The dataset of this project (OC2022 IS2RE) is obtained from here : [Open Catalyst Project](https://opencatalystproject.org/)

## Project Description

>**Note**  
>New 2DCNN approach description is at the bottom of the page!

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

### V. 2D Convolution Neural Network (2DConv.py)

The 3D CNN model requires relatively large space to store 1 training data. It turns out that only 500 data points take up more than 60GB of memory, which is not efficient.

The data of the 2D CNN model contains 3 adjacency matrices:

1. Distance between atoms.
2. Angles between atoms.
3. Diagonal matrix (atomic numbers).

The 2DConv.py code is designed for GPU computation, which means you need to install the CUDA toolkit and cuDNN in your environment. It is completely fine to train the model using only the CPU, but it will take more time.

The 2DConv_server.py code is intended for the server of our lab. You can run the code by using the command python 2DConv_server.py. If you want to edit the code, there are two ways:

1. Modify the code on your own computer and then upload it to the server.
2. Use the command vi 2DConv_server.py to edit the code directly in the command line interface.

>**Note**  
>Before using the GPU of server, you have to check the GPU status by `nvidia-smi`.
>We use the 4th GPU and the conda environment `AI`.

## Future Plans for the Project

The aforementioned predictive models are established as structure-energy forward models. The goal of this project is to train an accurate forward model using a large amount of density functional theory (DFT) data obtained from the Open Catalyst Project. Subsequently, the predicted energy values will be utilized as the reward mechanism in reinforcement learning.

Our ultimate objective is to infer the possible structure of a catalyst based on desired properties (in this project, the role of energy). We aim to implement this using reinforcement learning. The following is a brief overview of the process:

Reinforcement learning requires an environment to provide the agent with exploration opportunities, where the agent receives a reward at each visited state. In the context of the current structure-energy model, the environment represents atomic structures. The agent moves within different atomic structures, with each step representing a different structure. After each move, the agent utilizes the new structure as input to the forward model to predict its energy, which is then used as the reward.

For example, assuming the target energy is 500, but the predicted energy for a particular structure is 200, the reward might be -300. If the agent reaches a new structure and the predicted energy is 480, the reward could be set as -20. The simplest objective in reinforcement learning is to maximize the reward. Through multiple attempts, the agent gradually converges towards the desired energy.

Please note that the above explanation is a basic description of reinforcement learning and how it relates to the structure-energy model.
