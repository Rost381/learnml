import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import math
from ml.math_tools import mt


class SVM():
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0

        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]

        self.C = 1.0  # 松弛变量

    def kernel(self, x1, x2):
        """
        kernel 核函数
        """
        if self._kernel == 'linear':
            # return sum([x1[k] * x2[k] for k in range(self.n)])
            return np.dot(x1, x2.T)
        elif self._kernel == 'poly':
            # return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2
            return np.dot(x1, x2.T) ** 2
        else:
            return 0

    def _g(self, i):
        """
        g(x) = sum(a_j * y_j * K(x_i, x_j)) + b
        """
        g = self.b
        for j in range(self.m):
            g += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return g

    def _KKT(self, i):
        """
        KKT
        """
        yg = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:  # 边界内部
            return yg >= 1
        elif 0 < self.alpha[i] < self.C:  # 边界上
            return yg == 1
        else:  # 两条边界之间
            return yg <= 1

    def _E(self, i):
        """
        E(i) = g(x_i) - y_i
        g(x_i)对输入x_i的预测值和y_i的差
        """
        return self._g(i) - self.Y[i]

    def _init_alpha(self):
        """
        find alpha1, alpha2
        """
        # 0 < alpha < C
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # not satisfy 0 < alpha < C
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        # satisfy & not satisfy
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self._KKT(i):
                continue

            E1 = self.E[i]
            # if E1 >= 0，choose min
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            # if E1 < 0，choose max
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j

    def _compare(self, _alpha, L, H):
        """
        get alpha2_new
        """
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def fit(self, features, labels):
        self.init_args(features, labels)

        for each_iter in range(self.max_iter):
            # train
            i1, i2 = self._init_alpha()

            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]

            # eta = K11 + K22 - 2 * K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(
                self.X[i2], self.X[i2]) - 2 * self.kernel(self.X[i1], self.X[i2])
            if eta <= 0:
                continue

            """
            get new alpha1, alpha2
            """
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            alpha2_new = self._compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * \
                self.Y[i2] * (self.alpha[i2] - alpha2_new)

            """
            get new b1, b2
            """
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - \
                self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * \
                (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - \
                self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * \
                (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # choose middle
                b_new = (b1_new + b2_new) / 2

            """
            update alpha1, alpha2
            """
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        return 'train done!'

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        c = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                c += 1
        return c / len(X_test)
