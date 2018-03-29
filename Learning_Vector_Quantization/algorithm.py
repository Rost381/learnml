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

    r.sort(key = lambda x: x[1])

    return r[0][0]

def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])

    codebook = [train[randrange(n_records)][i] for i in range(n_features)]

    return codebook