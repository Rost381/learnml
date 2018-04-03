import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

test_row = dataset[0]

r = bmu(dataset, test_row)

print(r)

print(random_codebook(dataset))

n_codebooks = 2
lrate = 0.3
epochs = 10

codebooks = train_codebooks(dataset, n_codebooks, lrate, epochs)

print('Codebooks: %s' % codebooks)

for i in dataset:
    print(predict(codebooks, i))

# example 2
f = '../Data/ionosphere.csv'
d = pd.read_csv(f, header=None)
df = d.replace(['g', 'b'], [0, 1])
train, test = train_test_split(df, test_size=0.5)
total = test.shape[0]
train, test = train.values.tolist(), test.values.tolist()

n_codebooks = 20
lrate = 0.3
epochs = 50
n_folds = 5

codebooks = train_codebooks(train, n_codebooks, lrate, epochs)

n = 0

for i in test:
    p = predict(codebooks, i)
    a = i[-1]
    if p == a:
        n += 1
    #print('predict={0}, actual={1}'.format(p, a))

print('{:.2%}'.format(float(n / total)))
