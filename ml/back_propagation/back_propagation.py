import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import random, seed


import math
from ml.math_tools import mt


class BP():
    def str_column_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    def init_network(self, n_inputs, n_hidden, n_outputs):
        network = list()
        # hidden layer has [hidden] neuron with [inputs + 1] input weights plus the bias.
        hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]}
                        for i in range(n_hidden)]
        network.append(hidden_layer)
        # The output layer has [outputs] neurons, each with [hidden + 1] weight plus the bias.
        output_layer = [{'weights': [random() for i in range(n_hidden + 1)]}
                        for i in range(n_outputs)]
        network.append(output_layer)
        return network

    """
    step 1 forward_propagate
    """

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return 1.0 / (1.0 + math.exp(-activation))

    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            """
            * inputs = new_inputs, which is the last step value
            """
            inputs = new_inputs
        return inputs

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    """
    step 2 backward_propagate
    """

    def backward_propagate_error(self, network, expected):
        """
        store errors in each neuron
        error[i] = W * error[i+1] * sigmoid_derivative
        """
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()

            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])

            for j in range(len(layer)):
                neuron = layer[j]

                """
                * no output
                """
                neuron['delta'] = errors[j] * \
                    self.transfer_derivative(neuron['output'])

    """
    step 3 update_weights
    """

    def update_weights(self, network, row, lrate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]

            for neuron in network[i]:
                for j in range(len(inputs)):
                    """
                    * update weight
                    weight = weight + learning rate * error
                    because input = output, so use input here
                    """
                    neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]

                neuron['weights'][-1] += lrate * neuron['delta']

    def train_network(self, network, train, lrate, nepoch, noutputs):
        for epoch in range(nepoch):
            sum_error = 0

            for row in train:
                outputs = self.forward_propagate(network, row)  # cal output

                expected = [0 for i in range(noutputs)]

                expected[row[-1]] = 1

                sum_error += sum([(expected[i] - outputs[i]) **
                                  2 for i in range(len(expected))])

                self.backward_propagate_error(network, expected)  # get delta
                self.update_weights(network, row, lrate)

            print('epoch=%d, learning rate=%.3f, sum error=%.3f' %
                  (epoch, lrate, sum_error))

    def predict(self, network, row):
        outputs = self.forward_propagate(network, row)
        print(outputs)
        return outputs.index(max(outputs))
