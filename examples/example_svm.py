import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.support_vector_machine import support_vector_machine
from ml.math_tools import mt


def main():

    data = pd.read_csv('data/svm.csv', header=None)

    data = np.array(data)
    X, y = data[:, 0:-1], data[:, -1].astype(int)

    svm = support_vector_machine.SVM()

    support_vectors, iterations = svm.fit(X, y)

    sv_count = support_vectors.shape[0]
    print(sv_count)

    y_hat = svm.predict(X)

    print(svm.calc_acc(y, y_hat))


if __name__ == "__main__":
    main()
