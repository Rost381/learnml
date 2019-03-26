import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from alphalearn.api import Perceptron
from alphalearn.utils.api import (calculate_accuracy_score, cross_entropy_loss,
                                  l2_loss, normalize, to_categorical)


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4)

    model = Perceptron(n_iter=5000,
                       learning_rate=0.001,
                       loss=l2_loss)
    model.fit(X_train, y_train)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y = np.argmax(y_test, axis=1)

    accuracy = calculate_accuracy_score(y, y_pred)
    print("Accuracy Score: {:.2%}".format(accuracy))


if __name__ == "__main__":
    main()
