# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:07:22 2023

@author: Yu Chia Cheng
"""
import gym 
import keras
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

e_gpa = pd.read_excel('e_gpa_data.xlsx')
#%%


#ms_temp = pd.read_excel('ms_temp_data.xlsx')

e_tensile = e_gpa[e_gpa['method'] == 'tensile test']
e_tensile = e_tensile.drop(['method','No.', 'Ref.'], axis = 1)


#%%
#left, up, right, down
ACTIONS = [np.array([-1, 0]), 
           np.array([0, 1]),
           np.array([1, 0]),
           np.array([0, -1])]

gamma = 1
alpha = 0.1
epsilon = 0.3

S0 = [0, 0]

Q_0 = rd.rand(50, 6, 4)

def step(state, action):
  state = np.array(state)
  action = ACTIONS[action]
  state_n = (state + action).tolist()
  x, y = state_n
  
  if x < 0 or x >= 10 or y < 0 or y >= 5:
    reward = -1
    state_n = state.tolist()
  else:
    if y == 4:
      reward = -100
    else:
      reward = -1
  return state_n, reward

def chooseAction(Q, state, epsilon):

  p = rd.random()
  if p < epsilon:
    n = rd.randint(0,4)
  else:
    n = np.argmax(Q[state[0], state[1], :])
  return n

def rotate_matrix(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0])-1,-1,-1)]