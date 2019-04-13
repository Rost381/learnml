import numpy as np


class NeuralNetwork():
    def __init__(self, optimizer, loss, validation_data=None):
        self.optimzer = optimzer
        self.layer = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y": y}

    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        self.layers.append(layer)
