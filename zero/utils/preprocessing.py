from itertools import chain
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np


def PolynomialFeatures(X, degree):
    """Generate polynomial and interaction features.
    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
    Parameters
    ----------
    degree : integer
        The degree of the polynomial features. Default = 2.
    """
    n_samples, n_features = X.shape

    def _combinations(n_features, degree):
        comb = (combinations_w_r)
        return chain.from_iterable(comb(range(n_features), i)
                                   for i in range(0, degree + 1))
    combs = _combinations(n_features, degree)
    n_output_features_ = sum(1 for _ in combs)
    combs = _combinations(n_features, degree)
    XP = np.empty((n_samples, n_output_features_))

    for i, comb in enumerate(combs):
        XP[:, i] = X[:, comb].prod(1)
    return XP
