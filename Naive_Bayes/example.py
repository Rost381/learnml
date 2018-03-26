import math
import operator
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example 1
dataset = [[3.393533211, 2.331273381, 0],
           [3.110073483, 1.781539638, 0],
           [1.343808831, 3.368360954, 0],
           [3.582294042, 4.67917911, 0],
           [2.280362439, 2.866990263, 0],
           [7.423436942, 4.696522875, 1],
           [5.745051997, 3.533989803, 1],
           [9.172168622, 2.511101045, 1],
           [7.792783481, 3.424088941, 1],
           [7.939820817, 0.791637231, 1]]

print(summarize_dataset(dataset))

summary = summarize_class(dataset)
print(summary)
"""
mean(c), std(c), len(c) for column
{
0: [
    (2.7420144012, 0.9265683289298018, 5), 
    (3.0054686692, 1.1073295894898725, 5)
    ], 

1: [
    (7.6146523718, 1.2344321550313704, 5), 
    (2.9914679790000003, 1.4541931384601618, 5)
    ]
}
"""
probabilities = class_prob(summary, dataset[0])
print(probabilities)


# example 2
f = '../Data/iris.csv'
d = pd.read_csv(f, header=None)

mapping = {'Iris-virginica': 0, 'Iris-versicolor': 1, 'Iris-setosa': 2}

df = d.replace(['Iris-virginica', 'Iris-versicolor', 'Iris-setosa'], [0, 1, 2])

train, test = train_test_split(df, test_size=0.4)
train, test = train.values.tolist(), test.values.tolist()

summary = summarize_class(train)

m = 0
for i in test:
    prob = class_prob(summary, i)
    # print(p)
    p = max(prob.items(), key=operator.itemgetter(1))[0]
    t = i[-1]
    if p == t:
        m += 1

print(m / len(test))
