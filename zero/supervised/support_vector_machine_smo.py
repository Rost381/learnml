import math

import numpy as np
import pandas as pd

from zero.utils.kernels import linear, poly, rbf


class svmSMO():
    """ Support Vector Machine.
    Use Sequential minimal optimization (SMO) to solve the quadratic QP problem.
    When all the Lagrange multipliers satisfy the KKT, 
    the problem has been solved.

    Parameters:
    -----------
    C : float
        Penalty parameter C of the error term.
    kernel: string
        Specifies the kernel type to be used in the algorithm. It must be one of 'poly' or 'linear' or 'rbf'.
    max_iter: int
        The maximum number of iterations.
    degree : int
        Degree of the polynomial kernel function ('poly'). 
    gamma : float
        Kernel coefficient for 'rbf', 'poly'.
    coef0 : float
        Independent term in kernel function. It is only significant in 'poly'.
    """

    def __init__(self, C=1, max_iter=1000, kernel='linear', degree=3, gamma=None, coef0=0.0):
        self._C = C
        self._kernel_mapping = {
            'linear': linear,
            'poly': poly,
            'rbf': rbf
        }
        self._max_iter = max_iter
        self._kernel = kernel
        self._degree = degree
        self._gamma = gamma
        self._coef0 = coef0

        self.alpha = None
        self.E = None
        self.w = 0
        self.b = None

    def _g(self, i):
        """ g(x) = sum(a_j * y_j * K(x_i, x_j)) + b
        """
        g = self.b
        for j in range(self.m):
            g += self.alpha[j] * self.Y[j] * self._kernel(self.X[i], self.X[j])
        return g

    def _KKT(self, i):
        """ KKT
        """
        yg = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:  # inner
            return yg >= 1
        elif 0 < self.alpha[i] < self._C:  # border
            return yg == 1
        else:  # between the borders
            return yg <= 1

    def _E(self, i):
        """ E(i) = g(x_i) - y_i
        """
        return self._g(i) - self.Y[i]

    def _alpha_init(self):
        """ Find alpha1, alpha2
        """
        # 0 < alpha < C
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self._C]
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
        self.m, self.n = np.shape(X)
        self.X = X
        self.Y = y
        self.b = 0.0

        self._kernel = self._kernel_mapping[self._kernel](
            degree=self._degree,
            gamma=self._gamma,
            coef0=self._coef0)

        """ np.ones()
        np.ones(5) => array([ 1.,  1.,  1.,  1.,  1.])
        """
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]

        for each_iter in range(self._max_iter):
            i1, i2 = self._alpha_init()

            """ Caculate L, H
            """
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self._C)
                H = min(self._C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self._C, self._C + self.alpha[i2] - self.alpha[i1])

            """ Caculate E1, E2
            """
            E1 = self.E[i1]
            E2 = self.E[i2]

            """ Caculate eta
            eta = K11 + K22 - 2 * K12
            """
            eta = self._kernel(self.X[i1], self.X[i1]) + self._kernel(
                self.X[i2], self.X[i2]) - 2 * self._kernel(self.X[i1], self.X[i2])
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
            b1_new = -E1 - self.Y[i1] * self._kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - \
                self.Y[i2] * self._kernel(self.X[i2], self.X[i1]) * \
                (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self._kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - \
                self.Y[i2] * self._kernel(self.X[i2], self.X[i2]) * \
                (alpha2_new - self.alpha[i2]) + self.b

            """ Caculate new b
            """
            if 0 < alpha1_new < self._C:
                b_new = b1_new
            elif 0 < alpha2_new < self._C:
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

        self.w = sum(np.dot(self.alpha, self.X)) + self.b

        return None

    def predict(self, X_test):
        y_pred = []
        for sample in range(len(X_test)):
            prediction = self.b
            for i in range(self.m):
                prediction += self.alpha[i] * self.Y[i] * \
                    self._kernel(X_test[sample], self.X[i])

            """ np.sign()
            The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
            """
            y_pred_sample = np.sign(prediction)
            y_pred.append(y_pred_sample)
        return y_pred
