<div align="center">

![](docs/images/logo.png)

# zero
**Write machine learning algorithms from Zero**

![](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)
![](https://img.shields.io/badge/machine%20learning-algorithms-brightgreen.svg)
![](http://progressed.io/bar/20?)

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
[![image](https://img.shields.io/badge/Document-brightgreen.svg)](https://byzhi.github.io/zero/) here.

## Algorithms

### Supervised Learning
<p style="color:#959da5;">linear models</p>

- [**Regression**](./zero/supervised/regression.py)
  | [linear example](./examples/example_LinearRegression.py)
  | [lasso example](./examples/example_LassoRegression.py)
  | [ridge example](./examples/example_RidgeRegression.py)
  | [polynomial ridge example](./examples/example_PolynomialRidgeRegression.py)
- [**Logistic Regression**](./zero/supervised/logistic_regression.py) | [example](./examples/example_LogisticRegression.py)

<p style="color:#959da5;">classification</p>

- Perceptron
- [**Support Vector Machine**](./zero/supervised/support_vector_machine.py) | [example_01](./examples/example_svm.py) | [example_02](./examples/example_svm_02.py)
- k-nearest neighbor
- [**Linear Discriminant Analysis**](./zero/supervised/linear_discriminant_analysis.py) | [example](./examples/example_PCA_LDA.py)


<p style="color:#959da5;">tree-based and ensemble methods</p>

- [**Classification Tree**](./zero/supervised/decision_tree.py) | [example](./examples/example_ClassificationTree.py)
- [**RegressionTree**](./zero/supervised/decision_tree.py) | [example](./examples/example_RegressionTree.py)
- Random forests
- [**Adaboost**](./zero/supervised/adaboost.py) | [example](./examples/example_Adaboost.py)
- [**Gradient boosting**](./zero/supervised/gradient_boosting.py) | [example_classifier](./examples/example_GradientBoostingClassifier.py) | [example_regressor]()
- XGBoost


<p style="color:#959da5;">generative Learning</p>

- Naive Bayes

### Unsupervised Learning

<p style="color:#959da5;">dimension reduction</p>

-  [**Principal Component Analysis**](./zero/unsupervised/principal_component_analysis.py) | [example](./examples/example_PCA_LDA.py)
-  K-Means
-  FP-Growth

### Reinforcement Learning
- Q-learning

### Deep Learning
- DNN
- RNN
- CNN
