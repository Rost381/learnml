import numpy as np
import pandas as pd


def divide_on_feature(X, feature_i, threshold):
    """ Used for decision tree
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
