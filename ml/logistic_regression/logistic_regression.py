import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import random, seed


import math
from ml.math_tools import mt


class LOGIT():
    def predict(self, row, coef):
        yhat = coef[0]
        for i in range(len(row) - 1):
            yhat += coef[i + 1] * row[i]
        return 1.0 / (1.0 + math.exp(-yhat))

    def fit(self, train, l_rate, n_epoch):
        """
        Stochastic gradient descent
        """
        coef = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                yhat = self.predict(row, coef)
                error = row[-1] - yhat
                sum_error += error
                """
                equation from Formula (18.8) on page 727 of Artificial Intelligence a Modern Approach.
                """
                coef[0] = coef[0] + l_rate * error * yhat * (1 - yhat)
                for i in range(len(row) - 1):
                    """
                    Why do we not just update the current coefficient we are on? In other words, why do we do:
                    coef[i+1] = coef[i+1]…. etc
                    instead of doing
                    coef[i] = coef[i]…. etc

                    Because the coefficient at position 0 is the bias (intercept) coefficient which bumps the indices down one and misaligns them with the indices in the input data.
                    """
                    coef[i + 1] = coef[i + 1] + l_rate * \
                        error * yhat * (1.0 - yhat) * row[i]
        return coef
