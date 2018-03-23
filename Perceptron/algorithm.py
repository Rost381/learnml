# prediction
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i+1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# stochastic gradient descent
def sgd(dataset, l_rate, n_epoch):
    weights = [0.0 for i in range(len(dataset[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in dataset:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] + l_rate*error*row[i]
        #print('>epoch=%d, lrate=%.3f, error=%.3f, weights %s' % (epoch, l_rate, sum_error, weights))
    return weights