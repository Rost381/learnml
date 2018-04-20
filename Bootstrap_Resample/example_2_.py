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

train, test = train_test_split(df, test_size=0.5)
train, test = train.values.tolist(), test.values.tolist()

str_column_to_int(train, len(train[0]) - 1)
str_column_to_int(test, len(test[0]) - 1)

sample_size = 0.50
max_depth = 6
min_size = 2
sample = subsample(train, sample_size)



trees = tree_build(sample, max_depth, min_size)

for row in test:
    predictions = [predict(tree, row) for tree in trees]
    print(predictions)
