import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from learnml.datasets.api import load_banknote
from learnml.supervised.api import ClassificationTree
from learnml.utils.api import calculate_accuracy_score


def main():
    data = load_banknote()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4)

    model = ClassificationTree(max_depth=5)
    model.fit(X_train, y_train)
    model.print_tree()

    y_pred = model.predict(X_test)
    accuracy = calculate_accuracy_score(y_test, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))


if __name__ == "__main__":
    main()
