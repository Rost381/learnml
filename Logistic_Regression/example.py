import math
import os

import numpy as np
import pandas as pd

from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example 1
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

l_rate = 0.3
n_epoch = 100

coef = sgd(dataset, l_rate, n_epoch)
print(coef)

# example 2
f = '../Data/pima.csv'
d = pd.read_csv(f, header=None)

"""
mean normalization)
normalized_df=(df-df.mean())/df.std()

min-max normalization:
normalized_df=(df-df.min())/(df.max()-df.min())
"""
dn = (d - d.min()) / (d.max() - d.min())
print(dn)
dl = dn.values.tolist()

l_rate = 0.1
n_epoch = 100
coef = sgd(dl, l_rate, n_epoch)
print(coef)

i = 0
for row in d.values.tolist():
    yhat = predict(row, coef)
    if row[-1] == round(yhat):
        i += 1
print(i)

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
