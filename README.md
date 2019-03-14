# zero
**Write machine learning algorithms from Zero**

![](https://img.shields.io/badge/python-3.5+-blue.svg)
![](http://progressed.io/bar/18?)

## Features
- Friendly code.
- Implementation from Zero.
- Designed for easily learning algorithms.

## Getting started
Linear regression demo.

```python
from zero.api import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

## Document
[In progress..](https://byzhi.github.io/zero/) 

## Algorithms

### Supervised Learning
linear models

- [Regression](./zero/supervised/regression.py)
 | examples: [linear](./examples/example_LinearRegression.py)
, [lasso](./examples/example_LassoRegression.py)
, [ridge](./examples/example_RidgeRegression.py)
, [polynomial ridge](./examples/example_PolynomialRidgeRegression.py)
- [Logistic Regression](./zero/supervised/logistic_regression.py) | examples: [01](./examples/example_LogisticRegression.py)

classification

- Perceptron
- [Support Vector Machine](./zero/supervised/support_vector_machine.py) | examples: [01](./examples/example_svm.py), [02](./examples/example_svm_02.py)
- [k-Nearest Neighbors](./zero/supervised/k_nearest_neighbors.py) | examples: [01](./examples/example_KNeighborsClassifier.py)
- [Linear Discriminant Analysis](./zero/supervised/linear_discriminant_analysis.py) | examples: [01](./examples/example_PCA_LDA.py)


tree-based and ensemble methods

- [Classification Tree](./zero/supervised/decision_tree.py) | examples: [01](./examples/example_ClassificationTree.py)
- [RegressionTree](./zero/supervised/decision_tree.py) | examples: [01](./examples/example_RegressionTree.py)
- Random forests
- [Adaboost](./zero/supervised/adaboost.py) | examples: [01](./examples/example_Adaboost.py)
- [Gradient boosting](./zero/supervised/gradient_boosting.py) | examples: [classifier](./examples/example_GradientBoostingClassifier.py), [regressor](./examples/example_GradientBoostingRegressor.py)
- XGBoost

generative Learning

- Naive Bayes

### Unsupervised Learning

dimension reduction

- [Principal Component Analysis](./zero/unsupervised/principal_component_analysis.py) | examples: [01](./examples/example_PCA_LDA.py)
-  K-Means
-  FP-Growth

### Reinforcement Learning
- Q-learning

### Deep Learning
- DNN
- RNN
- CNN
