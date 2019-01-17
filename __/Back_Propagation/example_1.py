from algorithm import *

# example 1
network = init_network(2, 1, 2)

"""
1 * (first(2) + 1)
last(2) * (1 + 1)
"""
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
           [{'weights': [0.2550690257394217, 0.49543508709194095]},
            {'weights': [0.4494910647887381, 0.651592972722763]}]]

row = [1, 0]
output = forward_propagate(network, row)
print(output)


"""
loop 1:
    activation = 0.763774618976614

    activation = 1 * 0.13436424411240122 + 0 * 0.8474337369372327 + 0.763774618976614
    = 0.8981388630890152

    neuron['output'] = 1.0 / (1.0 + math.exp(-activation))
    = 0.7105668883115941

    inputs = [0.7105668883115941]

loop 2:
    loop 2-1:
        activation = 0.49543508709194095

        activation = inputs * 0.2550690257394217 + activation
        = 0.6766786910162718

        neuron['output'] = 1.0 / (1.0 + math.exp(-activation))
        = 0.6629970129852887

    loop 2-2:
        activation = 0.651592972722763

        activation = 0.7105668883115941 * 0.4494910647887381 + 0.651592972722763
        = 0.9709864399535617

        neuron['output'] = 1.0 / (1.0 + math.exp(-activation))
        = 0.7253160725279748

    inputs:[0.6629970129852887, 0.7253160725279748]

final:
[0.6629970129852887, 0.7253160725279748]
"""
