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

## Usage
Run exmaple_*.py under examples/ folder.

## Docs

## Mathematics and Python
Read theses simple code used in 'Zero' before learning algorithms.
* Mathematics
  * [Singular value decomposition(SVD)](./docs/singular_value_decomposition.py)
  * [Quadratic program](./docs/quadratic_program.py)
* Python
  * [Classes](./docs/python_class.py)
  * [Numpy](./docs/python_numpy.py)

## Algorithms
### Supervised Learning
* #### Regression

| Linear Regression | Lasso Regression | Ridge Regression |
| --- | --- | --- |
| Least squared error | L1 regularization| L2 regularization |
| SVD [[Code](./zero/supervised/regression.py)] [[Example](./examples/example_LinearRegression.py)] | Coordinate descent [[Paper](https://core.ac.uk/download/pdf/6287975.pdf)] [[Code](./zero/supervised/regression.py)] [[Example](./examples/example_LassoRegression.py)] | Gradient descent [[Code](./zero/supervised/regression.py)] [[Example](./examples/example_RidgeRegression.py)] |

* #### Support Vector Machine
    **Convex optimization** [[Code](./zero/supervised/support_vector_machine_cvxopt.py)] [[Example](./examples/example_svmCVXOPT.py)] | **SMO** [[Paper](https://pdfs.semanticscholar.org/59ee/e096b49d66f39891eb88a6c84cc89acba12d.pdf)] [[Code](./zero/supervised/support_vector_machine_smo.py)] [[Example](./examples/example_svmSMO.py)]

* #### Decision Trees

| Classification Tree | RegressionTree |
| --- | --- |
| Information gain [[Code](./zero/supervised/decision_tree.py)] [[Example](./examples/example_ClassificationTree.py)]| Variance reduction [[Code](./zero/supervised/decision_tree.py)] [[Example](./examples/example_RegressionTree.py)] |

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