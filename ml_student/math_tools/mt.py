import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


def covariance_matrix(X, Y=None):
    """
    Caculate covariance matrix
    """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * \
        (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    return np.array(covariance_matrix, dtype=float)


def calculate_eig(matrix):
    """
    Return eigenvalues, eigenvectors
    """
    return np.linalg.eig(matrix)


def calculate_variance(X):
    """
    Return the variance of the features in dataset X
    """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    return variance


def standardize(X):
    """
    standardize X
    """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    return X_std


def normalize(X, axis=-1, order=2):
    """
    normalize X
    """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def calculate_mean_squared_error(y_true, y_pred):
    """
    y_true:numpy.ndarray
    y_pred:list
    Returns the mean squared error between y_true and y_pred
    """
    mse = np.mean(np.power(y_true - y_pred, 2))
    #print("MSE: {0}".format(mse))
    return mse


def data_train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """
    Split the data into train and test sets
    """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def shuffle_data(X, y, seed=None):
    """
    Random shuffle of the samples in X and y
    """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def divide_on_feature(X, feature_i, threshold):
    """
    # decision tree
    Divide dataset based on if sample value on 
        feature index is larger than the given threshold
    """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        def split_func(sample): return sample[feature_i] >= threshold
    else:
        def split_func(sample): return sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


def calculate_entropy(y):
    """
    Calculate the entropy of label array y
    """
    def log2(x): return math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def calculate_accuracy_score(y_true, y_pred):
    """
    y_true:numpy.ndarray
    y_pred:list
    Compare y_true to y_pred and return the accuracy
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    # print("Accuracy Score: {:.2%}".format(accuracy))
    return accuracy
