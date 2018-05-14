import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from algorithm import *
from algorithm_tree import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))


f = '../Data/sonar.csv'
d = pd.read_csv(f, header=None)
df = d.replace(['R', 'M'], [0, 1])

n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(math.sqrt(len(df[0]) - 1))

train, test = train_test_split(df, test_size=0.5)
train, test = train.values.tolist(), test.values.tolist()

sample = subsample(train, sample_size)
tree = tree_build(sample, max_depth, min_size, n_features)