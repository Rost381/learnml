import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from alphalearn.deep.api import (Activation, AdamOptimizer, BatchNormalization,
                                 Conv2D, Dense, Dropout, Flatten,
                                 NeuralNetwork)
from alphalearn.utils.api import cross_entropy_loss, to_categorical


def main():
    optimizer = AdamOptimizer()

    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape((-1, 1, 8, 8))
    X_test = X_test.reshape((-1, 1, 8, 8))

    model = NeuralNetwork(optimizer=optimizer,
                          loss=cross_entropy_loss,
                          validation_data=(X_test, y_test))

    model.add(Conv2D(n_filters=16, filter_shape=(3, 3),
                     stride=1, input_shape=(1, 8, 8), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Conv2D(n_filters=32, filter_shape=(
        3, 3), stride=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary(name="ConvNet")

    train_err, val_err = model.fit(
        X_train, y_train, n_epochs=50, batch_size=256)

    _, accuracy = model.test_on_batch(X_test, y_test)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
