import copy

import numpy as np

from learnml.utils.api import (ELU, SELU, LeakyReLU, ReLU, Sigmoid, Softmax,
                               SoftPlus, TanH, column_to_image,
                               determine_padding, get_im2col_indices,
                               image_to_column)


class Layer():
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit,
                                   (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        W = self.W

        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units, )


class RNN(Layer):
    def __init__(self, n_units, activation='tanh', bptt_trunc=5, input_shape=None):
        self.n_units = n_units
        self.activation = activation_function[activation]()
        self.trainable = True
        self.bptt_trunc = bptt_trunc
        self.input_shape = input_shape
        """
        ---------------------------------------
        | W                  | V      | U     |
        ---------------------------------------
        | weight of previous | output | input |
        """
        self.W = None
        self.V = None
        self.U = None

    def initialize(self, optimzer):
        timesteps, input_dim = self.input_shape

        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))

        limit = 1 / math.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))

        limit = 1 / np.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))

        self.U_opt = copy.copy(optimizer)
        self.V_opt = copy.copy(optimizer)
        self.W_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.V.shape) + np.prod(self.U.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, timesteps, input_dim = X.shape

        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps + 1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        self.states[:, -1] = np.zeros((batch_size, self.n_units))

        for t in range(timesteps):
            self.state_input[:, t] = X[:, t].dot(
                self.U.T) + self.states[:, t - 1].dot(self.W.T)
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outpus[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward_pass(self, accum_grad):
        _, timesteps, _ = accum_grad.shape

        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)

        accum_grad_next = np.zeros_like(accum_grad)

        for t in reversed(range(timesteps)):
            # Update gradient w.r.t V at time step t
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = accum_grad[:, t].dot(
                self.V) * self.activation.gradient(self.state_input[:, t])
            # Gradient w.r.t the layer input
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t + 1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(
                    self.W) * self.activation.gradient(self.state_input[:, t_ - 1])

        # Update weights
        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

    def output_shape(self):
        return self.input_shape


class Conv2D(Layer):
    """2D Convolution Layer

    Parameters:
    -----------
    n_filters : int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape : tuple
        A tuple (filter_height, filter_width).
    input_shape : tuple
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
    padding : string
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
    stride : int
        The stride length of the filters during the convolution over the input.
    """

    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True

    def initialize(self, optimizer):
        # Initialize the weights
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / np.sqrt(np.prod(self.filter_shape))
        self.W = np.random.uniform(-limit, limit, size=(self.n_filters,
                                                        channels, filter_height, filter_width))
        self.w0 = np.zeros((self.n_filters, 1))
        # Weight optimizers
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        self.X_col = image_to_column(
            X, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        self.W_col = self.W.reshape((self.n_filters, -1))
        # Calculate output
        output = self.W_col.dot(self.X_col) + self.w0
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        return output.transpose(3, 0, 1, 2)

    def backward_pass(self, accum_grad):
        # Reshape accumulated gradient into column shape
        accum_grad = accum_grad.transpose(
            1, 2, 3, 0).reshape(self.n_filters, -1)

        if self.trainable:
            # Take dot product between column shaped accum. gradient and column shape
            # layer input to determine the gradient at the layer with respect to layer weights
            grad_w = accum_grad.dot(self.X_col.T).reshape(self.W.shape)
            # The gradient with respect to bias terms is the sum similarly to in Dense layer
            grad_w0 = np.sum(accum_grad, axis=1, keepdims=True)

            # Update the layers weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Recalculate the gradient which will be propogated back to prev. layer
        accum_grad = self.W_col.T.dot(accum_grad)
        # Reshape from column shape to image shape
        accum_grad = column_to_image(accum_grad,
                                     self.layer_input.shape,
                                     self.filter_shape,
                                     stride=self.stride,
                                     output_shape=self.padding)

        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(
            self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h)
                         - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w)
                        - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)


class Dropout(Layer):
    """A layer that randomly sets a fraction p of the output units of the previous layer
    to zero.

    Parameters:
    -----------
    p : float
        The probability that unit x is set to zero.
    """

    def __init__(self, p=0.2):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, X, training=True):
        c = (1 - self.p)
        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward_pass(self, accum_grad):
        return accum_grad * self._mask

    def output_shape(self):
        return self.input_shape


activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'selu': SELU,
    'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    'softplus': SoftPlus
}


class Activation(Layer):
    """A layer that applies an activation operation to the input.
    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class BatchNormalization(Layer):
    """Batch normalization"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.trainable = True
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None

    def initialize(self, optimizer):
        # Initialize the parameters
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        # parameter optimizers
        self.gamma_opt = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_pass(self, X, training=True):

        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = np.mean(X, axis=0)
            self.running_var = np.var(X, axis=0)

        if training and self.trainable:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        self.X_centered = X - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output

    def backward_pass(self, accum_grad):

        # Save parameters used during the forward pass
        gamma = self.gamma

        # If the layer is trainable the parameters are updated
        if self.trainable:
            X_norm = self.X_centered * self.stddev_inv
            grad_gamma = np.sum(accum_grad * X_norm, axis=0)
            grad_beta = np.sum(accum_grad, axis=0)

            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = accum_grad.shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        accum_grad = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * accum_grad -
            np.sum(accum_grad, axis=0) -
            self.X_centered * self.stddev_inv**2 *
            np.sum(accum_grad * self.X_centered, axis=0)
        )

        return accum_grad

    def output_shape(self):
        return self.input_shape


class Flatten(Layer):
    """Turns a multidimensional matrix into two-dimensional"""

    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return (np.prod(self.input_shape),)


class SimpleLstmLoss():
    """Computes square loss with first element of hidden layer array"""
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff
