# L1 Regularization

$$
\min J(W,b)=\frac{1}{m}\sum_{i=1}^mL(\hat{y}^i,y^i)+\frac{\lambda}{2m}||W^l||
$$

```python
return self.alpha * np.linalg.norm(w)
```

# L2 Regularization (weight decay)

$$
\min J(W,b)=\frac{1}{m}\sum_{i=1}^mL(\hat{y}^i,y^i)+\frac{\lambda}{2m}||W||_2^2
$$

```python
return self.alpha * 0.5 * w.T.dot(w)
```