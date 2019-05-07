import numpy as np

from learnml.datasets.api import load_iris
from learnml.supervised.api import AdaBoostClassifier
from learnml.utils.api import calculate_accuracy_score
from sklearn.datasets import make_gaussian_quantiles


def main():
    # Example 1
    def load_simple_data():
        features = ([
            [1.0, 2.1],
            [2.0, 1.1],
            [1.3, 1.0],
            [1.0, 1.0],
            [2.0, 1.0]
        ])
        labels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return np.array(features), np.array(labels)

    X, y = load_simple_data()
    model = AdaBoostClassifier(n_estimators=5)
    model.fit(X, y)

    y_pred = model.predict(X)
    print(y_pred)
    accuracy = calculate_accuracy_score(y, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))

    # Example 2
    X, y = make_gaussian_quantiles(n_samples=1300, n_features=10,
                                   n_classes=2)

    n_split = 300
    X_train, X_test = X[:n_split], X[n_split:]
    y_train, y_test = y[:n_split], y[n_split:]

    model = AdaBoostClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = calculate_accuracy_score(y_test, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))


if __name__ == "__main__":
    main()
