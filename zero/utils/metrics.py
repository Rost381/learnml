import numpy as np


def calculate_accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy
    y_true:numpy.ndarray
    y_pred:list
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    # print("Accuracy Score: {:.2%}".format(accuracy))
    return accuracy


def calculate_mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred
    y_true:numpy.ndarray
    y_pred:list
    """
    mse = np.mean(np.power(y_true - y_pred, 2))
    #print("MSE: {0}".format(mse))
    return mse
