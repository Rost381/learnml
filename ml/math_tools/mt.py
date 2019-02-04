import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


def data_train_test(df, test_size=0.4):
    """
    split data into train and test
    """
    train, test = train_test_split(df, test_size=test_size)
    test_total = test.shape[0]
    train, test = train.values.tolist(), test.values.tolist()
    return train, test, test_total


def covariance_matrix(X, Y=None):
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * \
        (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    return np.array(covariance_matrix, dtype=float)


def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)
