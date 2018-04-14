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
df = df.values.tolist()

max_depth = 6
min_size = 2
tree = tree_build(df, max_depth, min_size)
tree_print(tree)
