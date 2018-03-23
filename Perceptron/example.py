import os

import numpy as np
import pandas as pd

from algorithm import *
from sklearn.preprocessing import LabelEncoder
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example 1
dataset = [
    [2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]
]

l_rate = 0.1
n_epoch = 5
weights = sgd(dataset, l_rate, n_epoch)
print(weights)

# example 2
f = '../Data/sonar.csv'
d = pd.read_csv(f, header=None)
dl = d.values.tolist()
d = pd.to_numeric(dl, errors='coerce')
print(d)
#df = LabelEncoder().fit_transform(d)
#df = df.values.tolist()
#print(df)
'''
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

#for i in range(len(dl[0])-1):
#    str_column_to_float(dl, i)
str_column_to_int(dl, len(dl[0])-1)
'''

n_folds = 3
l_rate = 0.01
n_epoch = 500
weights = sgd(dl, l_rate, n_epoch)

print(weights)

i = 0
for row in dl:
    yhat = predict(row, weights)
    if row[-1] == round(yhat):
        i += 1
    #print("Expected=%3f, Predicted=%3f [%d]" % (row[-1], yhat, round(yhat)))

print(i)