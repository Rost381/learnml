import os
import sys
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_student.math_tools import mt


class LinearRegression():
    def _mean(self, values):
        """
        mean
        """
        return sum(values) / float(len(values))

    def _variance(self, values, mean):
        """
        variance
        """
        return sum([(x - mean)**2 for x in values])

    def _covariance(self, x, mean_x, y, mean_y):
        """
        covariance
        """
        covar = 0.0
        for i in range(len(x)):
            covar += (x[i] - mean_x) * (y[i] - mean_y)
        return covar

    def fit(self, dataset):
        r = self.coefficients(dataset)
        return r

    def coefficients(self, dataset):
        """
        coefficients
        """
        x = [row[0] for row in dataset]
        y = [row[1] for row in dataset]
        x_mean, y_mean = self._mean(x), self._mean(y)

        b1 = self._covariance(x, x_mean, y, y_mean) / self._variance(x, x_mean)
        b0 = y_mean - b1 * x_mean
        return [b0, b1]

    def predict(self, train, test):
        """
        predict
        """
        predctions = list()
        b0, b1 = self.coefficients(train)

        for row in test:
            yhat = b0 + b1 * row[0]
            predctions.append(yhat)

        return predctions
