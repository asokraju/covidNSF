"""
custom environment must follow the gym interface
skeleton copied from:
    https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
"""
import sys
sys.path.append("..")

import os

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import pandas as pd

from rpy2.robjects.packages import STAP

class SEIR_env(gym.Env):
  """
  Description:
        Each city's population is broken down into four compartments --
        Susceptible, Exposed, Infectious, and Removed -- to model the spread of
        COVID-19.

  Source:
        SEIR model from https://github.com/UW-THINKlab/SEIR/
        Code modeled after cartpole.py from
         github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
         
         
         
  Observation*:
        Type: Box(4,)
        Num     Observation       Min     Max
        0       Susceptible       0       Total Population
        1       Exposed           0       Total Population
        2       Infected          0       Total Population
        3       Recovered         0       Total Population
        
  
  Actions*:
        Type: Box(3,), min=-1 max=0
        Num     Action                                   Change in model
        0       reductions in workplace activity         affect transmission rate
        1       reductions in grocery shopping           affect transmission rate     
        2       rerductions in retail shopping           affect transmission rate
        

  Reward:
        reward = lambda * economic cost + (1-lambda) * public health cost
        
        Economic cost:
            sum(action)
        Health cost:
            -0.00001* number of infected
        lambda:
            a user defined weight. Default 0.5

  Episode Termination:
        Episode length (time) reaches specified maximum (end time)
        The end of analysis period is 120 days
  """


  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(SEIR_env, self).__init__()

    # get r Bayesian Updating
    # cwd = os.getcwd()
    # with open('/Users/benbernhard/Documents/GitHub/COVID19_RL/COVID19_models/SEIR/read_input.R', 'r') as f:
    #     string = f.read()
    # read_input = STAP(string, "read_input")
    # getData_r=read_input.getData

    # # Get input data
    # input_data = getData_r("/Users/benbernhard/Documents/GitHub/COVID19_RL/COVID19_models/SEIR/RL_input")
    # os.chdir(cwd)

    # SEIR model inputs
    #self.hospital_cap = hospitalCapacity
    
    # Read input
    input_path = '/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR'
    thetas_init = pd.read_csv(input_path + '/parameters.csv')
    thetas_sd_init = pd.read_csv(input_path + '/parameters_sd.csv')
    case_data = pd.read_csv(input_path + '/states.csv')
    reduction_time_series = pd.read_csv(input_path + '/time_series_predictions_daily.csv')
    reduction_time_series = reduction_time_series[reduction_time_series.columns[1:3]]/100
    self.daynum      = 0
    self.latent = 5.1
    self.gamma = 0.06
    
    #self.latent = thetas_init.iloc[0,6]
    #self.gamma = thetas_init.iloc[0,7]
    self.thetas_init = np.array(thetas_init.iloc[0,:6],dtype=np.float64)
    self.thetas_sd_init = np.array(thetas_sd_init.iloc[0,:6],dtype=np.float64)
    self.thetas_updated = self.thetas_init
    self.reduction_time_series = reduction_time_series
    self.reduction_time_curr = np.array(self.reduction_time_series.iloc[self.daynum,:],dtype = np.float64)
    #self.reduction_time_curr = np.array([[-0.2,-0.2]],dtype=np.float64)
    self.popu         = 100000
    self.current      = 1
    self.pred         = 1
    self.trainNoise   = False
    self.weight       = 0.8

    # Save intial conditions for reset
    self.St_data0 = case_data.iloc[0,0]
    self.Et_data0 = case_data.iloc[0,1]
    self.It_data0 = case_data.iloc[0,2]
    self.Rt_data0 = case_data.iloc[0,3]
    
    # Make spaces for the updated case numbers
    self.St_data = self.St_data0
    self.Et_data = self.Et_data0
    self.It_data = self.It_data0
    self.Rt_data = self.Rt_data0
    self.beta    = 0
    
    
    # Define action and observation space
    # They must be gym.spaces objects
    self.action_space = spaces.Box(low = -1, high = 1,shape = (3,), dtype = np.float64)
    self.observation_space = spaces.Box(0, np.inf,shape=(4,),dtype=np.float64)

    # random seed
    self.seed()

    # initialize state
    self.state  = np.empty(shape=(4,))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    # Check for valid action
    err_msg = "%r (%s) invalid" % (action, type(action))
    assert self.action_space.contains(action), err_msg

    # R <--> python conversions
    numpy2ri.activate() # automatic conversion of numpy objects to rpy2 objects

    # Update model based on actions
    action = (action-1)/2
    #reduction_factor = np.reshape(action,(self.num_cities,self.num_cities))

    # Get thetas updated for later calculations
    cwd = os.getcwd()
    input_path = '/Users/gracejia/Documents/A-UW/covid19 NSF Project/COVID19_RL/COVID19_models/SEIR'
    with open(input_path + '/update_thetas_weekly.R', 'r') as f:
        string = f.read()
    seir_theta = STAP(string, "seir_theta")
    seir_theta=seir_theta.getThetas
    
    modelOut = seir_theta(
                          thetas_so_far = self.thetas_init,
                          thetas_sd_so_far = self.thetas_sd_init,
                          pred = self.pred)
    
    os.chdir(cwd)
    self.thetas_updated = np.array(modelOut, dtype = np.float64)

    # Get the SEIR model from r script

    with open(input_path + '/seir_r_weekly.R', 'r') as f:
        string = f.read()
    seir_r_weekly = STAP(string, 'seir_r_weekly')
    seir_r_weekly = seir_r_weekly.seirPredictions
    
    statesOut = seir_r_weekly( reduction_control = action,
                              reduction_time_series = self.reduction_time_curr,
                              thetas_data = self.thetas_updated,
                              latent = self.latent,
                              gamma = self.gamma,
                              St_data = self.St_data,
                              Et_data = self.Et_data,
                              It_data = self.It_data,
                              Rt_data = self.Rt_data,
                              popu = self.popu,
                              current = self.current,
                              pred = self.pred
        )


    # Unpack output
    S  = statesOut[0][0]
    E  = statesOut[1][0]
    I  = statesOut[2][0]
    R  = statesOut[3][0]
    beta = statesOut[4][0]
    
    # Update state
    self.state = np.array((S,E,I,R))
    self.daynum += self.pred
    self.St_data = S
    self.Et_data = E
    self.It_data = I
    self.Rt_data = R
    self.beta    = beta
    
    # Print states 
    print('States:',self.state)
# =============================================================================
#     print(sum(self.state))
# =============================================================================
    print('Beta:',beta)
    print('Action picked:',action)

    # Reward
    economicCost = np.sum(action) + np.sum(self.reduction_time_curr)
    publichealthCost   = -0.00001*abs(self.It_data)
    
    reward = self.weight * economicCost + (1 - self.weight)* publichealthCost
    print('Reward:',reward)
    print(self.daynum)

    # Observation
    observation = np.reshape(self.state,(4,))

    # Check if episode is over
    done = bool(I < 0.5 or self.daynum >= 199)
    

    
    return observation, reward, done, {}

  def reset(self):

    # reset to initial conditions
    self.daynum = 0
    self.St_data = self.St_data0
    self.Et_data = self.Et_data0
    self.It_data = self.It_data0
    self.Rt_data = self.Rt_data0
    self.beta    = 0

    self.state = np.empty(shape=(4,))
    observation = np.reshape(self.state,(4,))

    return observation  # reward, done, info can't be included
