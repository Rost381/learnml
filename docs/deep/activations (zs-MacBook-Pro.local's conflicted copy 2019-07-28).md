# Activation function
In artificial neural networks, the activation function of a node defines the output of that node, or "neuron," given an input or set of inputs. This output is then used as input for the next node and so on until a desired solution to the original problem is found.

## Sigmoid

* Equation

$${f(x)={\frac {1}{1+e^{-x}}}}$$

```python
def __call__(self, x):
    return 1. / (1. + np.exp(-x))
```

* Derivative

$${f'(x)=f(x)(1-f(x))}$$

```python
return self.__call__(x) * (1 - self.__call__(x))
```

---

## Softmax

* Equation

$${f_{i}({\vec {x}})={\frac {e^{x_{i}}}{\sum _{j=1}^{J}e^{x_{j}}}}}$$

```python
ndim = np.ndim(x)
if ndim == 2:
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)
elif ndim > 2:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s
```

* Derivative

$$\frac{\partial f_{i}(\vec{x})}{\partial x_{j}}=f_{i}(\vec{x})\left(\delta_{i j}-f_{j}(\vec{x})\right)$$


```python
p = self.__call__(x)
return p * (1 - p)
```

---

## TanH

Hyperbolic tangent

* Equation

$${\tanh x={\frac {\sinh x}{\cosh x}}={\frac {e^{x}-e^{-x}}{e^{x}+e^{-x}}}={\frac {e^{2x}-1}{e^{2x}+1}}}$$

```python
return 2 / (1 + np.exp(-2 * x)) - 1
```

* Derivative

$$\dfrac{d\tanh}{dx}=1-\tanh^2$$

```python
return 1 - np.power(self.__call__(x), 2)
```

---

## ReLU
Rectified linear unit

* Equation

$${f(x)={\begin{cases}0&{\text{for }}x<0\\x&{\text{for }}x\geq 0\end{cases}}}$$

```python
return np.where(x >= 0, x, 0)
```

* Derivative

$${f'(x)={\begin{cases}0&{\text{for }}x<0\\1&{\text{for }}x\geq 0\end{cases}}}$$

```python
return np.where(x >= 0, 1, 0)
```

---

## LeakyReLU

Leaky rectified linear unit

* Equation

$${f(x)={\begin{cases}0.01x&{\text{for }}x<0\\x&{\text{for }}x\geq 0\end{cases}}}$$

```python
return np.where(x >= 0, x, self.alpha * x)
```

* Derivative

$${f'(x)={\begin{cases}0.01&{\text{for }}x<0\\1&{\text{for }}x\geq 0\end{cases}}}$$

```python
return np.where(x >= 0, 1, self.alpha)
```

---

## ELU
Exponential linear unit

* Equation

$${f(\alpha ,x)={\begin{cases}\alpha (e^{x}-1)&{\text{for }}x\leq 0\\x&{\text{for }}x>0\end{cases}}}$$

```python
return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))
```

* Derivative

$${f'(\alpha ,x)={\begin{cases}f(\alpha ,x)+\alpha &{\text{for }}x\leq 0\\1&{\text{for }}x>0\end{cases}}}$$

```python
return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)
```

---

## SELU

Scaled exponential linear unit

* Equation

$${f(\alpha ,x)=\lambda {\begin{cases}\alpha (e^{x}-1)&{\text{for }}x<0\\x&{\text{for }}x\geq 0\end{cases}}}$$

$$\lambda =1.0507, \alpha =1.67326$$

```python
self.alpha = 1.6732632423543772848170429916717
self.scale = 1.0507009873554804934193349852946
return self.scale * np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))
```

* Derivative

$${f'(\alpha ,x)=\lambda {\begin{cases}\alpha (e^{x})&{\text{for }}x<0\\1&{\text{for }}x\geq 0\end{cases}}}$$

```python
return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))
```

---

## SoftPlus

* Equation

$$f(x)=\ln \left(1+e^{x}\right)$$

```python
return np.log(1 + np.exp(x))
```

* Derivative

$${f'(x)={\frac {1}{1+e^{-x}}}}$$

```python
return 1 / (1 + np.exp(-x))
```