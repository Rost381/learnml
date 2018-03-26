import math


def separate_class(dataset):
    s = dict()
    for i in range(len(dataset)):
        v = dataset[i]
        c = v[-1]

        if c not in s:
            s[c] = list()

        s[c].append(v)
    return s


'''
for lable in s:
    print(lable)
    for row in s[lable]:
        print(row)
'''
# print(s)


def mean(x):
    return sum(x) / float(len(x))


def std(x):
    var = sum([(i - mean(x))**2 for i in x]) / float(len(x) - 1)
    return math.sqrt(var)


def summarize_dataset(dataset):
    """
    return column as list
    print([c for c in zip(*dataset)])
    """
    s = [(mean(c), std(c), len(c)) for c in zip(*dataset)]
    del(s[-1])
    return s


def summarize_class(dataset):
    sep = separate_class(dataset)
    summaries = dict()
    for c, row in sep.items():
        summaries[c] = summarize_dataset(row)
    return summaries


def gaussian_prob(x, mean, std):
    a = math.exp(-((x - mean)**2) / (2 * std**2))
    b = math.sqrt(2 * math.pi) * std
    return (1 / b) * a


def class_prob(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    prob = dict()

    for c_value, c_summaries in summaries.items():
        prob[c_value] = summaries[c_value][0][2] / float(total_rows)

        for i in range(len(c_summaries)):
            mean, std, count = c_summaries[i]
            prob[c_value] *= gaussian_prob(row[i], mean, std)

    return prob
