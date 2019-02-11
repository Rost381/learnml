import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml_student.linear_discriminant_analysis import linear_discriminant_analysis
from ml_student.math_tools import mt


def main():
    df = pd.read_csv("data/iris.csv")

    # convert 'setosa', 'virginica', 'versicolor' to 0, 1, 2
    df.species = pd.factorize(df.species)[0]

    X = df.iloc[:, :-1].values
    X = mt.normalize(np.array(X))
    y = df.iloc[:, -1]
    y = y.values

    lda = linear_discriminant_analysis.LDA()
    lda.plot_lda(X, y, 'example_lda.png')


if __name__ == "__main__":
    main()
