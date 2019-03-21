import numpy as np
from sklearn.model_selection import train_test_split

from alphalearn.api import GaussianNB
from alphalearn.datasets.api import load_iris
from alphalearn.utils.api import calculate_accuracy_score, normalize


def main():
    # Example 1
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])

    clf = GaussianNB()
    clf.fit(X, Y)
    print(clf.predict([[-0.8, -1]]))

    # Example 2
    iris = load_iris()
    X = normalize(iris.data)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = calculate_accuracy_score(y_test, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))


if __name__ == "__main__":
    main()
