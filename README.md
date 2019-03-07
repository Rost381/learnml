<div align="center">

![](docs/images/logo.png)

# zero
**Write machine learning algorithms from Zero**

![](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)
![](https://img.shields.io/badge/machine%20learning-algorithms-brightgreen.svg)
![](http://progressed.io/bar/20?)

</div>

## Features
- Friendly.
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
[Here.](https://byzhi.github.io/zero/) 

## Algorithms

### Supervised Learning
- linear models

  - [**Regression**](./zero/supervised/regression.py)
   | examples: [linear](./examples/example_LinearRegression.py)
  , [lasso](./examples/example_LassoRegression.py)
  , [ridge](./examples/example_RidgeRegression.py)
  , [polynomial ridge](./examples/example_PolynomialRidgeRegression.py)
  - [**Logistic Regression**](./zero/supervised/logistic_regression.py) | [example](./examples/example_LogisticRegression.py)

- classification

  - Perceptron
  - [**Support Vector Machine**](./zero/supervised/support_vector_machine.py) | examples: [01](./examples/example_svm.py), [02](./examples/example_svm_02.py)
  - k-nearest neighbor
  - [**Linear Discriminant Analysis**](./zero/supervised/linear_discriminant_analysis.py) | [example](./examples/example_PCA_LDA.py)


- tree-based and ensemble methods

  - [**Classification Tree**](./zero/supervised/decision_tree.py) | [example](./examples/example_ClassificationTree.py)
  - [**RegressionTree**](./zero/supervised/decision_tree.py) | [example](./examples/example_RegressionTree.py)
  - Random forests
  - [**Adaboost**](./zero/supervised/adaboost.py) | [example](./examples/example_Adaboost.py)
  - [**Gradient boosting**](./zero/supervised/gradient_boosting.py) | examples: [classifier](./examples/example_GradientBoostingClassifier.py), [regressor](./examples/example_GradientBoostingRegressor.py)
  - XGBoost

- generative Learning

  - Naive Bayes

### Unsupervised Learning

- dimension reduction

  - [**Principal Component Analysis**](./zero/unsupervised/principal_component_analysis.py) | [example](./examples/example_PCA_LDA.py)
  -  K-Means
  -  FP-Growth

### Reinforcement Learning
- Q-learning

### Deep Learning
- DNN
- RNN
- CNN
