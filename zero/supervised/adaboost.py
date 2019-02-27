import math

import numpy as np


class BestStump():
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None


class Adaboost():
    def __init__(self, max_iter=5):
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        w = np.full(n_samples, (1 / n_samples))
        # print(w)
        self.stumps = []

        for i in range(self.max_iter):
            stump = BestStump()

            min_error = float('inf')
            print(i)

            """ caculate error """
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                # print(feature_values)
                unique_values = np.unique(feature_values)
                # print("unique_values:{0}".format(unique_values))

                for threshold in unique_values:
                    p = 1
                    # print(np.shape(y))
                    prediction = np.ones(np.shape(y))
                    # print(prediction)
                    prediction[X[:, feature_i] < threshold] = -1

                    error = sum(w[y != prediction])

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        stump.polarity = p
                        stump.threshold = threshold
                        stump.feature_index = feature_i
                        min_error = error
                        print('<  threshold:{0}, feature_i:{1}'.format(
                            threshold, feature_i))
                    print(
                        '--- feature_index: {0}  threshold:{1} ---'.format(feature_i, threshold))
                print('error:{0}, min_error:{1}'.format(error, min_error))

            """ caculate alpha """
            stump.alpha = 0.5 * math.log((1 - min_error) / (min_error + 1e-10))
            print('stump.alpha:{0}'.format(stump.alpha))

            predictions = np.ones(np.shape(y))

            negative_idx = (
                stump.polarity * X[:, stump.feature_index] < stump.polarity * stump.threshold)
            print('stump.polarity:{0}  X[:, stump.feature_index]:{1}  stump.polarity:{2}  stump.threshold:{3}'.format(
                stump.polarity, X[:, stump.feature_index], stump.polarity, stump.threshold))
            print(negative_idx)

            predictions[negative_idx] = -1
            print('predictions: {0}'.format(predictions))

            """ update w """
            w *= np.exp(-stump.alpha * y * predictions)

            w /= np.sum(w)
            print('w: {0}:'.format(w))

            self.stumps.append(stump)

    def predict(self, X):
        n_samples = np.shape(X)[0]

        y_pred = np.zeros((n_samples, 1))
        print(self.stumps)
        for stump in self.stumps:
            predictions = np.ones(np.shape(y_pred))

            negative_idx = (
                stump.polarity * X[:, stump.feature_index] < stump.polarity * stump.threshold)

            predictions[negative_idx] = -1

            y_pred += stump.alpha * predictions

        y_pred = np.sign(y_pred).flatten()

        return y_pred
