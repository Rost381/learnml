"""Built-in activation functions.
Reference:
https://github.com/keras-team/keras/blob/master/keras/activations.py
https://en.wikipedia.org/wiki/Activation_function
"""
import numpy as np


def sigmoid(x):
    """Sigmoid activation function.
    """
    return 1. / (1. + np.exp(-x))


def softmax(x, axis=-1):
    """Softmax activation function.
    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = np.ndim(x)
    if ndim == 2:
        y = np.exp(x - np.max(x, axis, keepdims=True))
        return y / np.sum(y, axis, keepdims=True)
    elif ndim > 2:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        s = np.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)
