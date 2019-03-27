# alphalearn
write machine learning algorithms from for self-learning

![](https://img.shields.io/badge/python-3.5+-blue.svg)
![](http://progressed.io/bar/18?)

## Getting started
Linear regression

```python
from alphalearn.api import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

## Document
[docs](https://byzhi.github.io/alphalearn/) 

## Code

### Supervised Learning
**linear models**

- [Logistic Regression](./alphalearn/supervised/logistic_regression.py) | examples: [01](./examples/example_LogisticRegression.py)
- [Regression](./alphalearn/supervised/regression.py)
 | examples: [linear](./examples/example_LinearRegression.py)
, [lasso](./examples/example_LassoRegression.py)
, [ridge](./examples/example_RidgeRegression.py)
, [polynomial ridge](./examples/example_PolynomialRidgeRegression.py)

**classification**

- [Linear Discriminant Analysis](./alphalearn/supervised/linear_discriminant_analysis.py) | examples: [01](./examples/example_PCA_LDA.py)
- [k-Nearest Neighbors](./alphalearn/supervised/k_nearest_neighbors.py) | examples: [01](./examples/example_KNeighborsClassifier.py)
- [Perceptron](./alphalearn/supervised/perceptron.py) | examples: [01](./examples/example_Perceptron.py)
- [Support Vector Machine](./alphalearn/supervised/support_vector_machine.py) | examples: [01](./examples/example_svm.py), [02](./examples/example_svm_02.py)

**tree-based and ensemble methods**

- [Adaboost](./alphalearn/supervised/adaboost.py) | examples: [01](./examples/example_Adaboost.py)
- [Decision Tree](./alphalearn/supervised/decision_tree.py) | [classification](./examples/example_ClassificationTree.py), [regression](./examples/example_RegressionTree.py)
- [Gradient boosting](./alphalearn/supervised/gradient_boosting.py) | examples: [classifier](./examples/example_GradientBoostingClassifier.py), [regressor](./examples/example_GradientBoostingRegressor.py)
- [Random forests](./alphalearn/supervised/random_forest.py) | examples: [01](./examples/example_RandomForestClassifier.py)
- [XGBoost](./alphalearn/supervised/xgboost.py) | examples: [01](./examples/example_XGBoost.py)

**generative Learning**

- [Naive Bayes](./alphalearn/supervised/naive_bayes.py) | examples: [01](./examples/example_GaussianNB.py)

### Unsupervised Learning

**dimension reduction**

- [K-Means](./alphalearn/unsupervised/kmeans.py) | examples: [01](./examples/example_KMeans.py)
- [Principal Component Analysis](./alphalearn/unsupervised/principal_component_analysis.py) | examples: [01](./examples/example_PCA_LDA.py)
-  FP-Growth

### Reinforcement Learning
- Q-learning

### Deep Learning
- DNN
- RNN
- CNN

## Reference
Based on: 
- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [Lasso](https://github.com/satopirka/Lasso)
- [Implementing a Principal Component Analysis (PCA)](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)
- [Linear Discriminant Analysis Bit by Bit](https://sebastianraschka.com/Articles/2014_python_lda.html)