import numpy as np


class AdamOptimizer():
    """Adam optimizer.
    $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
    $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
    $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
    $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
    The sparse implementation of this algorithm does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). 
    Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).

    Parameters:
    -----------
    learning_rate : 
        A Tensor or a floating point value.  The learning rate.
    beta1 : 
        A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
    beta2 : 
        A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
    epsilon : 
        A small constant for numerical stability.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, w, g):
        self.m = np.zeros(np.shape(g))
        self.v = np.zeros(np.shape(g))

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(g, 2)

        m_t = self.m / (1 - self.beta1)
        v_t = self.v / (1 - self.beta2)

        self.w_ = self.learning_rate * m_t / (np.sqrt(v_t) + self.epsilon)

        return w - self.w_
