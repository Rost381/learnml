from algorithm import *

# example 2
network = [[{'output': 0.7105668883115941,  'weights': [0.13436424411240122,
                                                        0.8474337369372327, 0.763774618976614]}],
           [{'output': 0.6213859615555266,  'weights': [0.2550690257394217, 0.49543508709194095]},
            {'output': 0.6573693455986976,  'weights': [0.4494910647887381, 0.651592972722763]}]]

expected = [0, 1]
backward_propagate_error(network, expected)

for layer in network:
    print(layer)

"""
step 1:

len(network) = 2
reversed(range(len(network)) = [1, 0]

step 2: delta

i = 1
    loop:
        j = 0
        errors.append(0 - 0.6213859615555266); errors = [-0.6213859615555266]
        j = 1
        errors.append(1 - 0.6573693455986976); errors = [-0.6213859615555266, 0.34263065440130236]

    loop:
        j = 0
        neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
        neuron['delta'] = 0.34263065440130236 * transfer_derivative(0.6213859615555266)
        = -0.14619064683582808
        {'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': -0.14619064683582808}

        j = 1
        neuron['delta'] = 0.34263065440130236 * transfer_derivative(0.6573693455986976)
        = 0.0771723774346327
        {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': 0.0771723774346327}

step 3: errors

i != len(network) - 1

network[1]

i = 0
    loop:
        j = 0
        error += (neuron['weights'][j] * neuron['delta'])
        = 0.2550690257394217 * -0.14619064683582808 = -0.03728870586063054

        j = 1
        error += (neuron['weights'][j] * neuron['delta'])
        = 0.4494910647887381 * 0.0771723774346327 - 0.03728870586063054 = -0.0026004117552590952

        errors = [-0.0026004117552590952]

step 4: delta
    loop:
        j = 0
        neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
        -0.0026004117552590952 * transfer_derivative(0.7105668883115941)
        
        neuron['delta'] = -0.0005348048046610517
        
        [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': -0.0005348048046610517}]
        [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': -0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': 0.0771723774346327}]

"""


#print(0.34263065440130236 * transfer_derivative(0.8474337369372327) -0.12779522209001554)