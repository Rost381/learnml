# coding:UTF-8

import numpy as np
import algorithm as svm

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

f = '../Data/svm.csv'
d = pd.read_csv(f, header=None)

train = d[d.columns[:-1]]

label = d[d.columns[-1]]

b, alphas = smoSimple(train, label, 0.6, 0.001, 40)

print(b)
print(alphas[alphas > 0])
