import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import algorithm as svm
from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example 1
train = [[3, 3], [4, 3], [1, 1]]
label = [1, 1, -1]
b, alphas = simpleSMO(train, label, 0.6, 0.001, 1)

print(b)
print(alphas)

# example 2
f = '../Data/svm.csv'
d = pd.read_csv(f, header=None)

train = d[d.columns[:-1]]

label = d[d.columns[-1]]

b, alphas = simpleSMO(train, label, 0.6, 0.001, 1)

print(b)
print(alphas)
