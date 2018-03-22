from math import exp

# prediction
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))

# sgd
def sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            sum_error += error**2
            '''
            equation from Formula (18.8) on page 727 of Artificial Intelligence a Modern Approach.
            '''
            coef[0] = coef[0] + l_rate*error*yhat*(1-yhat)
            for i in range(len(row)-1):
                '''
                Why do we not just update the current coefficient we are on? In other words, why do we do:
                coef[i + 1] = coef[i+1]…. etc
                instead of doing
                coef[i] = coef[i]…. etc

                Because the coefficient at position 0 is the bias (intercept) coefficient which bumps the indices down one and misaligns them with the indices in the input data.
                '''
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef