#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:18:04 2020

@author: gracejia
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd

#%%
import sys
sys.path.append('..')
import os
from rpy2.robjects.packages import STAP
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

from rpy2.robjects import pandas2ri
#%%
# cwd = os.getcwd()
# with open('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/read_input.R', 'r') as f:
#     string = f.read()
# read_input = STAP(string, "read_input")
# getData_r=read_input.getData

# # Get input data
# input_data = getData_r("/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR")
# os.chdir(cwd)

# # Unpack input data
# thetas_init = np.array(input_data.rx2("thetas_init")[0:6],dtype=np.float64)
# thetas_sd_init = np.array(input_data.rx2("thetas_sd_init")[0:6],dtype=np.float64)
# latent = input_data.rx2("latent")[0]
# gamma = input_data.rx2("gamma")[0]
# states = input_data.rx2("states")[0:4]
# reduction_time_series = input_data.rx2("time_series_predictions")






#%%
reduction_time_series = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/time_series_predictions.csv')
reduction_control = np.array([[-0.2,-0.2,-0.2]],dtype=np.float64)
reduction_time_series = reduction_time_series[reduction_time_series.columns[0:2]]
reduction_time_series = np.array(reduction_time_series.iloc[0,:], dtype = np.float64)
#%%
# initial conditions derived from May 3 to May 9
thetas_init = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/parameters.csv')
thetas_sd_init = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/parameters_sd.csv')
case_data = pd.read_csv('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/states.csv')

thetas_init = np.array(thetas_init.iloc[0,:6],dtype=np.float64)
thetas_sd_init = np.array(thetas_sd_init.iloc[0,:6],dtype=np.float64)
#%%
#thetas_updated = np.zeros((2,6))
numpy2ri.activate()
pandas2ri.activate()

cwd = os.getcwd()
with open('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/update_thetas_weekly.R', 'r') as f:
    string = f.read()
seir_theta = STAP(string, "seir_theta")
seir_theta=seir_theta.getThetas

modelOut = seir_theta(
                      thetas_so_far = thetas_init,
                      thetas_sd_so_far = thetas_sd_init,
                      pred = 1)

os.chdir(cwd)
print(modelOut)
#%%
with open('/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR/seir_r_weekly.R', 'r') as f:
    string = f.read()
seir_r_weekly = STAP(string, "seir_r_weekly")
seir_r_weekly = seir_r_weekly.seirPredictions

# Plug in SEIR model
statesOut = seir_r_weekly(reduction_control = reduction_control,
                         reduction_time_series = reduction_time_series,
                         thetas_data    = np.array(modelOut,dtype = np.float64),
                         latent       = 1,
                         gamma        = 0.33,
                         St_data      = case_data.iloc[0,0],
                         Et_data      = case_data.iloc[0,1],
                         It_data      = case_data.iloc[0,2],
                         Rt_data      = case_data.iloc[0,3],
                         popu         = 100000,
                         current      = 1,
                         pred         = 1)
                  
os.chdir(cwd)

# Unpack output

print(statesOut)
