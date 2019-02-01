import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from ml.decision_tree import decision_tree
from ml.math_tools import mt


def main():
    dataset = np.array([['青年', '否', '否', '一般', '否'],
                        ['青年', '否', '否', '好', '否'],
                        ['青年', '是', '否', '好', '是'],
                        ['青年', '是', '是', '一般', '是'],
                        ['青年', '否', '否', '一般', '否'],
                        ['中年', '否', '否', '一般', '否'],
                        ['中年', '否', '否', '好', '否'],
                        ['中年', '是', '是', '好', '是'],
                        ['中年', '否', '是', '非常好', '是'],
                        ['中年', '否', '是', '非常好', '是'],
                        ['老年', '否', '是', '非常好', '是'],
                        ['老年', '否', '是', '好', '是'],
                        ['老年', '是', '否', '好', '是'],
                        ['老年', '是', '否', '非常好', '是'],
                        ['老年', '否', '否', '一般', '否'],
                        ['青年', '否', '否', '一般', '是']])

    train_data = pd.DataFrame(dataset)

    dt = decision_tree.DT()
    print(dt)


if __name__ == "__main__":
    main()
