import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example 1
d = [
    [1.47, 52.21],
    [1.5, 53.12],
    [1.52, 54.48],
    [1.55, 55.84],
    [1.57, 57.2],
    [1.6, 58.57],
    [1.63, 59.93],
    [1.65, 61.29],
    [1.68, 63.11],
    [1.7, 64.47],
    [1.73, 66.28],
    [1.75, 68.1],
    [1.78, 69.92],
    [1.8, 72.19],
    [1.83, 74.46]
]

b0, b1 = coefficients(d)
print('b0={0}, b1={1}'.format(b0, b1))

# example 2
f = '../Data/insurance.csv'
d = pd.read_csv(f, header=None)
train, test = train_test_split(d, test_size=0.4)
train, test = train.values.tolist(), test.values.tolist()

b0, b1 = coefficients(train)

p = predict(train, test)
a = [row[-1] for row in test]
r = rmse(a, p)

print('b0={0}, b1={1}, rmse={2}'.format(b0, b1, r))
