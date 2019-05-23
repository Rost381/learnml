import numpy as np


def random_subsets(X, y, n_subsets, replacements=True):
    """Return random subsets"""
    n_samples = np.shape(X)[0]
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])
    return subsets


def random_arr(a, b, *args):
    """createst uniform random array w/ values in [a,b) and shape args"""
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a