import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.linear_discriminant_analysis import linear_discriminant_analysis
from ml.math_tools import mt

from sklearn import datasets


def main():
    df = pd.read_csv("data/iris.csv")
    X_ = df[df.columns[:-1]]
    X = mt.normalize(np.array(X_))
    y_ = df[df.columns[-1]].replace(to_replace=['setosa',
                                                'virginica', 'versicolor'], value=[0, 1, 2])
    y = np.array(y_)

    lda = linear_discriminant_analysis.LDA()
    lda.plot_lda(X, y, title="LDA")


if __name__ == "__main__":
    main()
