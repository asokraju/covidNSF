#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:26:06 2020

@author: gracejia
"""

import sys
sys.path.append("..")

import os

import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

from rpy2.robjects import pandas2ri
#from robjects.robjects.conversion import localconverter

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from rpy2.robjects.packages import STAP

# Pseudo parameters to put in seir_r
population = 100000
num_of_weeks = 25
city_name = 'Seattle'
reduction_control = np.array([[-0.2,-0.2,-0.2]])
reduction_control = np.repeat(reduction_control,num_of_days, axis = 0)
reduction_time_series = np.zeros((num_of_days,2))
latent = 5.1
gamma = 0.05
beta = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/beta.csv')
beta_sd = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/beta_sd.csv')
St_data = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/S.csv')
Et_data = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/E.csv')
It_data = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/I.csv')
Rt_data = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/R.csv')



for i in [St_data,Et_data,It_data,Rt_data, beta , beta_sd]:
    i = np.array(i[city_name])

numpy2ri.activate()
pandas2ri.activate()

# Get model SEIR model for the numbers in the S E I R compartments
cwd = os.getcwd()
with open('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/seir_r_weekly.R', 'r') as f:
    string = f.read()
seir_r_weekly = STAP(string, "seir_r_weekly")
seir_r_weekly = seir_r_weekly.seirPredictions

# Plug in SEIR model
modelOut = seir_r_weekly(reduction_control = reduction_control,
                         reduction_time_series = reduction_time_series,
                         thetas_data    = thetas_data,
                         latent       = latent,
                         gamma        = gamma,
                         St_data      = St_data,
                         Et_data      = Et_data,
                         It_data      = It_data,
                         Rt_data      = Rt_data,
                         popu         = population,
                         current      = 0,
                         pred         = 0 + num_of_days)
                  
os.chdir(cwd)

# Unpack output
print(1)
print(modelOut)