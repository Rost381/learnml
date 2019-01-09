import os
import sys
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.math_tools import mt


class LR():

    # mean
    def mean(self, values):
        return sum(values) / float(len(values))

    # variance
    def variance(self, values, mean):
        return sum([(x - mean)**2 for x in values])

    # covariance
    def covariance(self, x, mean_x, y, mean_y):
        covar = 0.0
        for i in range(len(x)):
            covar += (x[i] - mean_x) * (y[i] - mean_y)
        return covar

    # coefficients
    def coefficients(self, dataset):
        x = [row[0] for row in dataset]
        y = [row[1] for row in dataset]
        x_mean, y_mean = self.mean(x), self.mean(y)

        b1 = self.covariance(x, x_mean, y, y_mean) / self.variance(x, x_mean)
        b0 = y_mean - b1 * x_mean
        return [b0, b1]

    # predict
    def predict(self, train, test):
        predctions = list()
        b0, b1 = self.coefficients(train)

        for row in test:
            yhat = b0 + b1 * row[0]
            predctions.append(yhat)

        return predctions

    # Mean Squared Error
    def mse(self, actual, predicted):
        sum_error = 0.0
        
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
            mean_error = sum_error / float(len(actual))
        return mean_error
