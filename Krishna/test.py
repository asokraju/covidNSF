import numpy as np
from collections import deque
import yaml
import pickle
import os
from datetime import datetime
from shutil import copyfile
import random
import numbers
import functools
import operator

def load_config(filename):
    """Load and return a config file."""

    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

path = 'Krishna\config.yaml'
data = load_config(path)
for key in data.keys():
    print(key, ':', data[key])