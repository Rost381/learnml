import numpy as np


class Adam():
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, w, grad_wrt_w):
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_wrt_w
        self.v = self.beta2 * self.v + \
            (1 - self.beta2) * np.power(grad_wrt_w, 2)

        m_t = self.m / (1 - self.beta1)
        v_t = self.v / (1 - self.beta2)

        self.w_updt = self.learning_rate * m_t / (np.sqrt(v_t) + self.epsilon)

        return w - self.w_updt
