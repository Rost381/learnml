import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.support_vector_machine import support_vector_machine
from ml.math_tools import mt


def main():

    # example 1
    train = [[3, 3], [4, 3], [1, 1]]
    label = [1, 1, -1]

    svm = support_vector_machine.SVM()
    b, alphas = svm.simpleSMO(train, label, 1, 0.001, 10000)

    print(b)
    print(alphas)


if __name__ == "__main__":
    main()
