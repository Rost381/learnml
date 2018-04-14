from algorithm import *

dataset = [[randrange(10)] for i in range(20)]
print(dataset)

ratio = 0.10

for size in [1, 10, 100]:
    sample_means = list()
    for i in range(size):
        sample = subsample(dataset, ratio)
        print(sample)
        sample_mean = mean([row[0] for row in sample])
        sample_means.append(sample_mean)

    print('Samples=%d, Estimated Mean: %.3f' % (size, mean(sample_means)))
