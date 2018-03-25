def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

def sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            #print('row: %s' % row)
            #print('coef: %s' % coef)

            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error**2
            coef[0] = coef[0] - l_rate * error

            #print('yhat: %s' % yhat)
            #print('row[-1] %s' % row[-1])
            #print('error = yhat - row[-1]: %s' % error)
            #print('coef[0] = coef[0] - l_rate * error: %s' % coef[0])
            #print('-' * 10)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
                #print('-' * 10)
        #print("epoch=%d, lrate=%.3f, error=%.3f" % (epoch, l_rate, sum_error))
    return coef