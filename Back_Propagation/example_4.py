import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example 4


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


f = '../Data/seeds.csv'
d = pd.read_csv(f, header=None)

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
df = min_max_scaler.fit_transform(d)

str_column_to_int(df, len(df[0]) - 1)

train, test = train_test_split(df, test_size=0.5)

train, test = train.tolist(), test.tolist()
print(train)
str_column_to_int(train, len(train[0]) - 1)
str_column_to_int(test, len(test[0]) - 1)

n_inputs = len(train[0]) - 1
n_outputs = len(set([row[-1] for row in train]))
l_rate = 0.3
n_epoch = 500
n_hidden = 5

network = init_network(n_inputs, n_hidden, n_outputs)

train_network(network, train, l_rate, n_epoch, n_outputs)

for row in test:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
