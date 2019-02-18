import cvxopt
import numpy as np
import pandas as pd

from mlalgo.utils.kernels import linear_kernel, polynomial_kernel, rbf_kernel

cvxopt.solvers.options['show_progress'] = False


class svmCVXOPT():

    def __init__(self, C=1, kernel=linear_kernel, power=4, gamma=None, coef=4, tol=1e-4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.tol = tol
        self.lagr_multipliers = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        if not self.gamma:
            self.gamma = 1 / n_features

        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        kernel_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        """ cvxopt.matrix
        'i', 'd', and 'z', for integer, real (double), and complex matrices, respectively.
        """
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)

        G = cvxopt.matrix(np.identity(n_samples) * -1)
        h = cvxopt.matrix(np.zeros(n_samples))

        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        lagr_mult = np.ravel(solution['x'])
        idx = lagr_mult > self.tol

        """ Get the corresponding lagr. multipliers, support vectors
        """
        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors_ = X[idx]
        self.support_vector_labels_ = y[idx]

        """ Caculate intercept
        
        self.intercept_ = self.support_vector_labels_[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept_ -= self.lagr_multipliers[i] * self.support_vector_labels_[
                i] * self.kernel(self.support_vectors_[i], self.support_vectors_[0])
        """

        """ Caculate weight
        """
        alphas = np.array(solution['x'])
        self.coef_ = np.array((np.sum(alphas * y[:, None] * X, axis=0)))
        self.intercept_ = (self.support_vector_labels_ -
                           np.dot(self.support_vectors_, self.coef_))[0]

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels_[
                    i] * self.kernel(self.support_vectors_[i], sample)
            prediction += self.intercept_
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
