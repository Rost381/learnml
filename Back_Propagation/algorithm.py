from random import random, seed
import math


def init_network(n_inputs, n_hidden, n_outputs):
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


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        # print('weights[-1]:{0}'.format(weights[-1]))
        # print('weight[i]:{0}'.format(weights[i]))
        # print('inputs[i]:{0}'.format(inputs[i]))
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
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])

            #print('activation:{0}'.format(activation))
            #print('neuron[\'output\']:{0}'.format(neuron['output']))
            #print('-' * 10)
        inputs = new_inputs
        #print('inputs:{0}'.format(inputs))
    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                    '''
                    print('neuron[\'weights\'][j]:{0}'.format(neuron['weights'][j]))
                    print('neuron[\'delta\']:{0}'.format(neuron['delta']))
                    print('error:{0}'.format(error))'''
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
                #print('errors:{0}'.format(errors))

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
            '''
            print('----')
            print(errors)
            print(errors[j])
            
            print('neuron[\'delta\']:{0}'.format(neuron['delta']))'''


def update_weights(network, row, lrate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        
        for neuron in network[i]:
            for j in range(len(inputs)):
                # weight = weight + learning rate * error * input
                neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]

            neuron['weights'][-1] += lrate * neuron['delta']


def train_network(network, train, lrate, nepoch, noutputs):
    for epoch in range(nepoch):
        sum_error = 0
        
        for row in train:
            outputs = forward_propagate(network, row) # cal output
            #print('output:{0}'.format(outputs))
            
            expected = [0 for i in range(noutputs)]

            expected[row[-1]] = 1
            #print('expected:{0}'.format(expected))
            
            sum_error += sum([(expected[i] - outputs[i]) **
                              2 for i in range(len(expected))])

            backward_propagate_error(network, expected) # get delta
            
            #print('row:{0}'.format(row))
            #print('network:{0}'.format(network))
            update_weights(network, row, lrate)
            #

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lrate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
