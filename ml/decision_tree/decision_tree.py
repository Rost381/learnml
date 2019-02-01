import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import math
# from ml.math_tools import mt


class Node:
    def __init__(self, data=None, label=None, x=None, y=None):
        self.data = data
        self.label = label
        self.x = x
        self.y = y
        self.child = []

    def append(self, node):
        self.child.append(node)


class DT():
    def __init__(self, epsilon=0, alpha=0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.tree = Node()

    def prob(self, dataset):
        datalen = len(dataset)
        print(datalen)
        labelx = set(dataset)
        print(labelx)

        p = {l: 0 for l in labelx}
        print(p)
        for d in dataset:
            p[d] += 1
        for i in p.items():
            print(i)
            p[i[0]] /= datalen
        return p

    def entropy_calc(self, dataset):
        p = self.prob(dataset)
        ent = sum([-v * math.log(v, 2) for v in p.values])
        return ent

    def entropy_cond(self, dataset, col):
        labelx = set(datasets.iloc[col])
        p = {x: [] for x in labelx}
        for i, d in enumerate(dataset.iloc[-1]):
            p[dataset.iloc[col[i]]].append(d)
        return sum([self.prob(dataset.iloc[col])[k]*self.entropy_calc(p[k]) for k in p.keys()])
    
