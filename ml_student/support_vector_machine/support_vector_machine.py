import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import math
from ml_student.math_tools import mt


class SVM():
    """ Support Vector Machine.

    Use Sequential minimal optimization (SMO) to solve the quadratic QP problem.
    When all the Lagrange multipliers satisfy the KKT, 
    the problem has been solved.

    Args:
        max_iter: int
            limit on iterations within solver.
        kernel: string
            polynomial or linear.
        C: float
            Penalty term.
    """

    def __init__(self, max_iter=100, kernel='linear', C=1.0):
        self.max_iter = max_iter
        self._kernel = kernel
        self.m = None
        self.n = None
        self.X = None
        self.Y = None
        self.b = None
        self.alpha = None
        self.E = None
        self.C = C

    def kernel(self, x1, x2):
        """ Kernel
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
        """ g(x) = sum(a_j * y_j * K(x_i, x_j)) + b
        """
        g = self.b
        for j in range(self.m):
            g += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return g

    def _KKT(self, i):
        """ KKT
        """
        yg = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:  # 边界内部
            return yg >= 1
        elif 0 < self.alpha[i] < self.C:  # 边界上
            return yg == 1
        else:  # 两条边界之间
            return yg <= 1

    def _E(self, i):
        """ E(i) = g(x_i) - y_i
        """
        return self._g(i) - self.Y[i]

    def _alpha_init(self):
        """ Find alpha1, alpha2
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

    def _alpha_L_H(self, _alpha, L, H):
        """ Calculate alpha2_new
        """
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def fit(self, X, y):
        #self._init_args(features, labels)

        self.m, self.n = np.shape(X)
        self.X = X
        self.Y = y
        self.b = 0.0

        """ np.ones()
        np.ones(5) => array([ 1.,  1.,  1.,  1.,  1.])
        """
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]

        for each_iter in range(self.max_iter):
            i1, i2 = self._alpha_init()

            """ Caculate L, H
            """
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            """ Caculate E1, E2
            """
            E1 = self.E[i1]
            E2 = self.E[i2]

            """ Caculate eta
            eta = K11 + K22 - 2 * K12
            """
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(
                self.X[i2], self.X[i2]) - 2 * self.kernel(self.X[i1], self.X[i2])
            if eta <= 0:
                continue

            """ Caculate new alpha2(first), alpha1
            """
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            alpha2_new = self._alpha_L_H(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * \
                self.Y[i2] * (self.alpha[i2] - alpha2_new)

            """ Caculate new b1, b2
            """
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - \
                self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * \
                (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - \
                self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * \
                (alpha2_new - self.alpha[i2]) + self.b

            """ Caculate new b
            """
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            """ Update alpha1, alpha2, b
            """
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            """ Update E
            """
            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        return None

    def predict(self, X_test):
        y_pred = []
        for sample in range(len(X_test)):
            prediction = self.b
            for i in range(self.m):
                prediction += self.alpha[i] * self.Y[i] * \
                    self.kernel(X_test[sample], self.X[i])

            """ np.sign
            The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
            """
            y_pred_sample = np.sign(prediction)
            y_pred.append(y_pred_sample)
        return y_pred
