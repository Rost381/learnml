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
#df = df.values.tolist()
train, test = train_test_split(df, test_size=0.5)
train, test = train.values.tolist(), test.values.tolist()
'''
for i in range(len(train[0])-1):
    str_column_to_float(train, i)
for i in range(len(test[0])-1):
    str_column_to_float(test, i)

str_column_to_int(train, len(train[0]) - 1)
str_column_to_int(test, len(test[0]) - 1)
'''
max_depth = 6
min_size = 2
sample_size = 0.50
#tree = tree_build(df, max_depth, min_size)
# tree_print(tree)

train = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1]]

test = [
           [2.999208922, 2.209014212, 0],
           [6.642287351, 3.319983761, 1]]

for n_trees in [1, 5, 10, 50]:
    predicted = bagging(train, test, max_depth, min_size, sample_size, n_trees)
    
