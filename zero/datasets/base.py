import csv
import os
import sys
from os.path import dirname, join

import numpy as np
import pandas as pd

from zero.utils.bunch import Bunch


def load_data(module_path, data_file_name):
    """ Loads data from module_path/data/data_file_name. """
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names


def load_iris():
    """ Load and return the iris dataset. """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'iris.csv')

    return Bunch(data=data,
                 target=target,
                 target_names=target_names)


def load_banknote():
    """ Load and return the banknote dataset. """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'banknote.csv')

    return Bunch(data=data, target=target,
                 target_names=target_names)


def load_boston():
    """ Load and return the boston house-prices dataset. """
    module_path = dirname(__file__)

    data_file_name = join(module_path, 'data', 'boston_house_prices.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

    return Bunch(data=data,
                 target=target)


def load_temperature():
    """ Load and return the banknote dataset. """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'temperature.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        data = np.empty((n_samples))
        target = np.empty((n_samples,))

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[0], dtype=np.float64)
            target[i] = np.asarray(d[1], dtype=np.float64)

        return Bunch(data=data,
                     target=target)
