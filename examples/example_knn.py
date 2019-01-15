import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.k_nearest_neighbors import k_nearest_neighbors
from ml.math_tools import mt


def main():
    knn = k_nearest_neighbors.KNN()

    df_ = pd.read_csv("data/abalone.csv", header=None)
    df = df_.replace(['F', 'I', 'M'], [0, 1, 2])
    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = df[cols]

    train, test = train_test_split(df, test_size=0.4)
    total = test.shape[0]
    train, test = train.values.tolist(), test.values.tolist()

    right_count = 0
    for test_row in test:
        predict_value = knn.predict(train, test_row, 5)
        actual_value = test_row[-1]
        if predict_value == actual_value:
            right_count += 1
    print('{:.2%}'.format(float(right_count / total)))


if __name__ == "__main__":
    main()
