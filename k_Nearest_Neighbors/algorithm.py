import math


def edistance(x1, x2):
    r = 0.0
    for i in range(len(x1) - 1):
        r += (x1[i] - x2[i]) ** 2
    return math.sqrt(r)


def get_neighbors(train, test_row, k):
    r = list()

    for train_row in train:
        d = edistance(train_row, test_row)
        r.append((train_row, d))

    r.sort(key=lambda x: x[1])

    n = list()

    for i in range(k):
        n.append(r[i][0])

    return n


def predict(train, test_row, k):

    n = get_neighbors(train, test_row, k)

    o = [i[-1] for i in n]

    m = max(o, key=o.count)

    return m
