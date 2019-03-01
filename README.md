<div align="center">

![](docs/logo.png)

# zero
> Write Machine learning algorithms from Zero

![](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)
![](https://img.shields.io/badge/machine%20learning-algorithms-brightgreen.svg)
![](http://progressed.io/bar/25?)

</div>

## Features
- Friendly code and document.
- Minimal implementation from Zero.
- Designed for easily learning algorithms.

## Getting started
Linear regression demo.
```python
from zero.api import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

## Document
[Here](./docs/README.md).

## Algorithms

### Supervised Learning
-  [**Regression**](./zero/supervised/regression.py)
  ([Linear Example](./examples/example_LinearRegression.py))
  ([Lasso Example](./examples/example_LassoRegression.py))
  ([Ridge Example](./examples/example_RidgeRegression.py))
  ([Polynomial Ridge Example](./examples/example_PolynomialRidgeRegression.py))

- [**Logistic Regression**](./zero/supervised/logistic_regression.py) ([Example](./examples/example_LogisticRegression.py))

-  [**Support Vector Machine**](./zero/supervised/support_vector_machine.py) ([Example 01](./examples/example_svm.py)) ([Example 02](./examples/example_svm_02.py))

-  [**Classification Tree**](./zero/supervised/decision_tree.py) ([Example](./examples/example_ClassificationTree.py))
-  [**RegressionTree**](./zero/supervised/decision_tree.py) ([Example](./examples/example_RegressionTree.py))

- [**Linear Discriminant Analysis**](./zero/supervised/linear_discriminant_analysis.py) ([Example](./examples/example_PCA_LDA.py))

- [**Adaboost**](./zero/supervised/adaboost.py) ([Example](./examples/example_Adaboost.py))

### Unsupervised Learning
-  [**Principal Component Analysis**](./zero/unsupervised/principal_component_analysis.py) ([Example](./examples/example_PCA_LDA.py))

### Reinforcement Learning
* in progress...

### Deep Learning
* in progress...
