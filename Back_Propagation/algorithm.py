from random import random, seed
import math


def init_network(inputs, hidden, outputs):
    network = list()
    """
    hidden layer has [hidden] neuron with [inputs + 1] input weights plus the bias. 
    """
    hidden_layer = [{'weight': [random() for i in range(inputs + 1)]}
                    for i in range(hidden)]

    network.append(hidden_layer)

    """
    The output layer has [outputs] neurons, each with [hidden + 1] weight plus the bias.
    """
    output_layer = [{'weight': [random() for i in range(hidden + 1)]}
                    for i in range(outputs)]

    network.append(output_layer)

    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        print('weights[-1]:{0}'.format(weights[-1]))
        print('weight[i]:{0}'.format(weights[i]))
        print('inputs[i]:{0}'.format(inputs[i]))
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            print('activation:{0}'.format(activation))
            print('-'*10)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs