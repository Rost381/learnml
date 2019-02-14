import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml_student.k_nearest_neighbors import k_nearest_neighbors
from ml_student.math_tools import mt


def main():
    model = k_nearest_neighbors.KNN()

    df = pd.read_csv("data/abalone.csv", header=None)
    df_ = df.replace(['F', 'I', 'M'], [0, 1, 2])
    cols = df_.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = df_[cols]

    train, test = train_test_split(df, test_size=0.4)
    total = test.shape[0]
    train, test = train.values.tolist(), test.values.tolist()

    y_pred = []
    y_test = []
    for row in test:
        y_pred.append(model.predict(train, row, 5))
        y_test.append(row[-1])

    accuracy = mt.calculate_accuracy_score(y_test, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))


if __name__ == "__main__":
    main()
