from random import random, seed


def init_network(inputs, hidden, outputs):
    network = list()

    hidden_layer = [{'weight': [random() for i in range(inputs + 1)]}
                    for i in range(hidden)]
    print(hidden_layer)
    network.append(hidden_layer)

    output_layer = [{'weight': [random() for i in range(hidden + 1)]}
                    for i in range(outputs)]

    network.append(output_layer)

    return network


network = init_network(2, 1, 2)

print(network)
