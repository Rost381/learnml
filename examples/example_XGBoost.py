import numpy as np
from sklearn.model_selection import train_test_split

from learnml.datasets.api import load_iris
from learnml.supervised.api import XGBoost
from learnml.utils.api import calculate_accuracy_score


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    model = XGBoost(n_estimators=300, learning_rate=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = calculate_accuracy_score(y_test, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))


if __name__ == "__main__":
    main()
