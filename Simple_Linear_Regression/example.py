import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example 1
d = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]

b0, b1 = coefficients(d)
print('b0={0}, b1={1}'.format(b0, b1))

# example 2
f = '../data/insurance.csv'
d = pd.read_csv(f, header=None)
train, test = train_test_split(d, test_size=0.4)
train, test = train.values.tolist(), test.values.tolist()

b0, b1 = coefficients(train)

p = predict(train, test)
a = [row[-1] for row in test]
r = rmse(a, p)

print('b0={0}, b1={1}, rmse={2}'.format(b0, b1, r))
