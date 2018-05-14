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

filename = '../Data/sonar.csv'
dataset = load_csv(filename)
print(type(dataset))
# convert string attributes to integers
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

folds = cross_validation_split(dataset, 5)
for fold in folds:
    train = list(folds)
    train.remove(fold)
    train = sum(train, [])
    test = list()
    for row in fold:
        row_copy = list(row)
        test.append(row_copy)
        row_copy[-1] = None

for n_trees in [1, 5, 10, 50]:
    predicted = bagging(train, test, max_depth, min_size, sample_size, n_trees)
    
