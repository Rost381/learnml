import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import random, seed

import math
from ml.math_tools import mt


class KNN():
    def _edistance(self, x1, x2):
        r = 0.0
        for i in range(len(x1) - 1):
            r += (x1[i] - x2[i]) ** 2
        return math.sqrt(r)

    def _get_neighbors(self, train, test_row, k):
        r = list()

        for train_row in train:
            d = self._edistance(train_row, test_row)
            r.append((train_row, d))
            
        r.sort(key=lambda x: x[1])

        n = list()
        for i in range(k):
            n.append(r[i][0])

        return n

    def predict(self, train, test_row, k):

        n = self._get_neighbors(train, test_row, k)
        o = [i[-1] for i in n]
        m = max(o, key=o.count)

        return m
