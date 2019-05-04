import cvxopt
import numpy as np

from alphalearn.utils.kernels import linear, poly, rbf

cvxopt.solvers.options['show_progress'] = False


class svm():
    """Support Vector Machine.

    Use cvxopt to solve the quadratic QP problem.

    Parameters:
    -----------
    C : float
        Penalty parameter C of the error term.
    kernel: string
        Specifies the kernel type to be used in the algorithm. It must be one of 'poly' or 'linear' or 'rbf'.
    degree : int
        Degree of the polynomial kernel function ('poly'). 
    gamma : float
        Kernel coefficient for 'rbf', 'poly'.
    coef0 : float
        Independent term in kernel function. It is only significant in 'poly'.
    tol : float
        Tolerance for stopping criterion.

    Attributes:
    -----------
    support_vectors_ : array-like
        Support vectors.
    coef_ : array
        Weights assigned to the features (coefficients in the primal problem).
        This is only available in the case of a linear kernel.
    intercept_ : array
        Constants in decision function.
        Best hyperplane in Linear SVM:
        y = ax + b
        a = -coef_[0] / coef_[1]
        b = -intercept_ / coef_[1]
    """

    def __init__(self, C=1, kernel='linear', degree=3, gamma=None, coef0=0.0, tol=1e-3):
        self._C = C
        self._kernel_mapping = {
            'linear': linear,
            'poly': poly,
            'rbf': rbf
        }
        self._kernel = kernel
        self._degree = degree
        self._gamma = gamma
        self._coef0 = coef0
        self._tol = tol
        self._lagr_multipliers = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = np.shape(X)

        if not self._gamma:
            self._gamma = 1 / n_features

        self._kernel = self._kernel_mapping[self._kernel](
            degree=self._degree,
            gamma=self._gamma,
            coef0=self._coef0)

        kernel_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self._kernel(X[i], X[j])

        """Cvxopt.matrix
        'i', 'd', and 'z', for integer, real (double), and complex matrices, respectively.
        """
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)

        if not self._C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            """numpy.identity
            The identity array is a square array with ones on the main diagonal.

            np.identity(3)
            array([[ 1.,  0.,  0.],
                [ 0.,  1.,  0.],
                [ 0.,  0.,  1.]])
            """
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)

            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self._C)

            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        """Solves a quadratic program
        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b.
        Arguments
        ---------
        P is a n x n dense or sparse 'd' matrix with the lower triangular
        part of P stored in the lower triangle.  Must be positive
        semidefinite.
        q is an n x 1 dense 'd' matrix.
        G is an m x n dense or sparse 'd' matrix.
        h is an m x 1 dense 'd' matrix.
        A is a p x n dense or sparse 'd' matrix.
        b is a p x 1 dense 'd' matrix or None.

        The default values for G, h, A and b are empty matrices with
        zero rows.
        """
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        lagr_mult = np.ravel(solution['x'])
        idx = lagr_mult > self._tol

        """Get the corresponding lagr. multipliers, support vectors"""
        self._lagr_multipliers = lagr_mult[idx]
        self.support_vectors_ = X[idx]
        self.support_vector_labels_ = y[idx]

        """Caculate intercept"""
        self.intercept_ = self.support_vector_labels_[0]

        for i in range(len(self._lagr_multipliers)):
            self.intercept_ -= self._lagr_multipliers[i] * self.support_vector_labels_[
                i] * self._kernel(self.support_vectors_[i], self.support_vectors_[0])

        """Caculate weight"""
        alphas = np.array(solution['x'])
        self.coef_ = np.array((np.sum(alphas * y[:, None] * X, axis=0)))

        """
        self.intercept_ = (self.support_vector_labels_ -
                           np.dot(self.support_vectors_, self.coef_))
        """

    def predict(self, X):
        y_pred = []
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self._lagr_multipliers)):
                prediction += self._lagr_multipliers[i] * self.support_vector_labels_[
                    i] * self._kernel(self.support_vectors_[i], sample)
            prediction += self.intercept_
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
