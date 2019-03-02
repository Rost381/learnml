# AdaBoost
An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

## Algorithms (Discrete AdaBoost)

### Initialize the weights ${\displaystyle w_{1,1}\dots w_{n,1}}$ set to ${\displaystyle {\frac {1}{n}}}$

```python
w = np.full(n_samples, (1 / n_samples))
```
**while** $m \leq M$ **do**

### Calculate minimum error
$$
\operatorname { err } _ { m } = \sum _ { i = 1 } ^ { N } w _ { i } ^ { ( m ) } h \left( - y _ { i } H _ { m } \left( \mathbf { x } _ { \mathbf { i } } \right) \right)
$$

```python
error = sum(w[y != prediction])
```

### Calcuate alpha

Minimize

$$\sum_{i}w_{i}e^{-y_{i}h_{i}\alpha_{t}}$$

Using the convexity of the exponential function

$$
\sum _ { i } w _ { i } e ^ { - y _ { i } h _ { i } \alpha _ { t } } \leq \sum _ { i } \left( \frac { 1 - y _ { i } h _ { i } } { 2 } \right) w _ { i } e ^ { \alpha _ { t } } + \sum _ { i } \left( \frac { 1 + y _ { i } h _ { i } } { 2 } \right) w _ { i } e ^ { - \alpha _ { t } }
$$

$$
= \left( \frac { \epsilon _ { t } } { 2 } \right) e ^ { \alpha _ { t } } + \left( \frac { 1 - \epsilon _ { t } } { 2 } \right) e ^ { - \alpha _ { t } }
$$

Set it to zero to find the minimum

$$
\left( \frac { \epsilon _ { t } } { 2 } \right) e ^ { \alpha _ { t } } - \left( \frac { 1 - \epsilon _ { t } } { 2 } \right) e ^ { - \alpha _ { t } } = 0
$$

So

$$
\alpha _ { m } = \frac { 1 } { 2 } \log \left( \frac { 1 - \operatorname { err } _ { m } } { \operatorname { err } _ { m } } \right)
$$

```python
stump.alpha = 0.5 * math.log((1 - error_min) / (error_min + 1e-6))
```
For each sample i = 1,...,N, update the weight

$$
w _ { i } ^ { ( m ) } = w _ { i } ^ { ( m ) } \exp \left( - \alpha _ { m } y _ { i } H _ { m } \left( \mathbf { x } _ { \mathbf { i } } \right) \right)
$$

```python
w *= np.exp(-stump.alpha * y * predictions)
```

### Renormalize the weights: $S _ { m } = \sum _ { j = 1 } ^ { N } v _ { j }$ and, for $i = 1 , \dots , N$

$$
w _ { i } ^ { ( m + 1 ) } = v _ { i } ^ { ( m ) } / S _ { m }$$

```python
w /= np.sum(w)
```

$
m \leftarrow m + 1
$

**end while**

### Predict
$$
H ( \mathbf { x } ) = \operatorname { sign } \left( \sum _ { j = 1 } ^ { M } \alpha _ { j } H _ { j } ( \mathbf { x } ) \right)
$$

```python
y_pred += stump.alpha * self.H(X, stump, y_init)
y_pred = np.sign(y_pred).flatten()
```