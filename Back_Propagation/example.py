from algorithm import *

# example 1
network = init_network(2, 1, 2)
# print(network)

network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
           [{'weights': [0.2550690257394217, 0.49543508709194095]},
            {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)

# example 2
network = [[{'output': 0.7105668883115941,  'weights': [0.13436424411240122,
                                                        0.8474337369372327, 0.763774618976614]}],
           [{'output': 0.6213859615555266,  'weights': [0.2550690257394217, 0.49543508709194095]},
            {'output': 0.6573693455986976,  'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)

for layer in network:
    print(layer)

# example 3
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

n_inputs = len(dataset[0]) - 1
print(n_inputs)
n_outputs = len(set([row[-1] for row in dataset]))

network = init_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)

for layer in network:
    print(layer)

for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))