# Machine learning algorithms

## Getting Started
Minimal implementation of machine learning algorithms from Zero. This project was designed for easily learning algorithms. For readability, python source code are all with friendly comments.

## Installation
```
$ git clone https://github.com/byzhi/zero
$ cd zero
$ python setup.py install
```
## Demo
```python
from zero.api import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

## Docs
Read theses simple code used in 'Zero' before learning algorithms.
* Mathematics
  * [Singular value decomposition(SVD)](./docs/singular_value_decomposition.py)
  * [Quadratic program](./docs/quadratic_program.py)
* Python
  * [Classes](./docs/python_class.py)
  * [Numpy](./docs/python_numpy.py)

## Algorithms
### Supervised Learning
- **Regression** [[Code](./zero/supervised/regression.py)]
  * Linear Regression  [[Example](./examples/example_LinearRegression.py)]
  * Lasso Regression [[Example](./examples/example_LassoRegression.py)]
  * Ridge Regression [[Example](./examples/example_RidgeRegression.py)]

- **Support Vector Machine** [[Code](./zero/supervised/support_vector_machine_cvxopt.py)] [[Example](./examples/example_svmCVXOPT.py)] 

- **Decision Trees**
  * Classification Tree [[Code](./zero/supervised/decision_tree.py)] [[Example](./examples/example_ClassificationTree.py)]
  * RegressionTree [[Code](./zero/supervised/decision_tree.py)] [[Example](./examples/example_RegressionTree.py)]

- **Linear Discriminant Analysis** [[Code](./zero/supervised/linear_discriminant_analysis.py)] [[Example](./examples/example_PCA_LDA.py)]

---

### Unsupervised Learning
- **Principal Component Analysis** [[Code](./zero/unsupervised/principal_component_analysis.py)]  [[Example](./examples/example_PCA_LDA.py)]


---

### Reinforcement Learning
* in progress...

---

### Deep Learning
* in progress...