from math import sqrt

# mean 期望
def mean(values):
    return sum(values) / float(len(values))

# variance 方差
def variance(values, mean):
    return sum([(x - mean)**2 for x in values])


# covariance 协方差
'''
如果协方差为正，说明X，Y同向变化，协方差越大说明同向程度越高；
如果协方差为负，说明X，Y反向运动，协方差越小说明反向程度越高。
'''
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


# 相关系数
'''
X、Y的协方差除以X的标准差和Y的标准差。

那为何要对它做平方呢？因为有时候变量值与均值是反向偏离的，是个负数，平方后，就可以把负号消除了。
这样在后面求平均时，每一项数值才不会被正负抵消掉，最后求出的平均值才能更好的体现出每次变化偏离均值的情况。

值越接近1，说明两个变量正相关性（线性）越强，越接近-1，说明负相关性越强，当为0时表示两个变量没有相关性。
'''

# coefficients 系数
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)

    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

# predict
def predict(train, test):
    predctions = list()
    b0, b1 = coefficients(train)

    for row in test:
        yhat = b0 + b1 * row[0]
        predctions.append(yhat)

    return predctions


# Root Mean Squared Error 方均根差
def rmse(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)
