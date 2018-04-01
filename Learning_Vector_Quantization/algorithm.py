import math
from random import randrange


def edistance(x1, x2):
    r = 0.0
    for i in range(len(x1) - 1):
        r += (x1[i] - x2[i]) ** 2
    return math.sqrt(r)


def bmu(codebooks, test_row):
    r = list()

    for codebook in codebooks:
        d = edistance(codebook, test_row)
        r.append((codebook, d))

    r.sort(key=lambda x: x[1])

    return r[0][0]


"""
random col value from train[0],[1],[2], respectively
"""


def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])

    codebook = [train[randrange(n_records)][i] for i in range(n_features)]

    return codebook


def train_codebooks(train, n_codebooks, lrate, epochs):

    # count(n_codebooks) of date
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    print(codebooks)
    for epoch in range(epochs):
        rate = lrate * (1.0 - (epoch / float(epochs)))
        sum_error = 0.0
        for row in train:
            bmu_value = bmu(codebooks, row)
            for i in range(len(row) - 1):
                error = row[i] - bmu_value[i]
                sum_error += error**2
                if bmu_value[-1] == row[-1]:
                    bmu_value[i] += rate * error
                else:
                    bmu_value[i] -= rate * error

        print('> epoch=%d, rate=%.3f, sum error=%.3f' % (epoch, rate, sum_error))
        print(codebooks)

    return codebooks
