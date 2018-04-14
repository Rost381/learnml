from random import seed
from random import random
from random import randrange


def subsample(dataset, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def mean(numbers):
    return sum(numbers) / float(len(numbers))
