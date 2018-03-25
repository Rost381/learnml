import math
import os

import numpy as np
import pandas as pd

from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example 1
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
l_rate = 0.001
n_epoch = 50
coef = sgd(dataset, l_rate, n_epoch)
print(coef)

# example 2
f = '../Data/wine.csv'
d = pd.read_csv(f, header=None)

dn = (d - d.min()) / (d.max() - d.min())
dl = dn.values.tolist()

n_folds = 5
l_rate = 0.01
n_epoch = 50
coef = sgd(dl, l_rate, n_epoch)
print(coef)

a = [row[-1] for row in dl]
p = [predict(row, coef) for row in dl]


def rmse(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error**2)
        mean_error = sum_error / float(len(actual))
    return math.sqrt(mean_error)


print(rmse(a, p))
