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

row0 = dataset[0]

for row in dataset:
    distance = edistance(row0, row)
    print(distance)

neighbors = get_neighbors(dataset, dataset[0], 3)
print(neighbors)

prediction = predict(dataset, dataset[0], 3)
print(prediction)

# example 2
f = '../Data/abalone.csv'
d = pd.read_csv(f, header=True)

df = d.replace(['F', 'I', 'M'], [0, 1, 2])

# order
cols = df.columns.tolist()
cols = cols[1:] + cols[:1]
df = df[cols]

train, test = train_test_split(df, test_size=0.5)

# count of row
total = test.shape[0]

train, test = train.values.tolist(), test.values.tolist()

k = 5
n = 0

for test_row in test:
    p = predict(train, test_row, k)
    a = test_row[-1]
    if p == a:
        n += 1

print('{:.2%}'.format(float(n / total)))
