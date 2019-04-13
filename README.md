# Alphalearn
![](https://img.shields.io/badge/python-3.7+-blue.svg)

Write machine learning algorithms from zero for self-learning. 

All algorithm codes are implemented in Python with friendly comments and easier to understand how they works.

## Installation
```
git clone https://github.com/byzhi/alphalearn
cd alphalearn
pip install -e .
```

## Usage
```python
from alphalearn.api import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

## Document
[docs](https://byzhi.github.io/alphalearn/) 

## Algorithms

### Supervised Learning
Linear models

- [Logistic Regression](./alphalearn/supervised/logistic_regression.py)
- [Regression](./alphalearn/supervised/regression.py)

Classification

- [Linear Discriminant Analysis](./alphalearn/supervised/linear_discriminant_analysis.py)
- [k-Nearest Neighbors](./alphalearn/supervised/k_nearest_neighbors.py)
- [Perceptron](./alphalearn/supervised/perceptron.py)
- [Support Vector Machine](./alphalearn/supervised/support_vector_machine.py)

Tree-based and ensemble methods

- [Adaboost](./alphalearn/supervised/adaboost.py)
- [Decision Tree](./alphalearn/supervised/decision_tree.py)
- [Gradient boosting](./alphalearn/supervised/gradient_boosting.py)
- [Random forests](./alphalearn/supervised/random_forest.py)
- [XGBoost](./alphalearn/supervised/xgboost.py)

Generative Learning

- [Naive Bayes](./alphalearn/supervised/naive_bayes.py)

### Unsupervised Learning

Dimension reduction

- [K-Means](./alphalearn/unsupervised/kmeans.py)
- [Principal Component Analysis](./alphalearn/unsupervised/principal_component_analysis.py)
-  FP-Growth

### Reinforcement Learning
- [Q-learning](./alphalearn/reinforcement/qlsarsa/base.py)
- [SARSA](./alphalearn/reinforcement/qlsarsa/base.py)
- [DQN (Deep Q Network)](./alphalearn/reinforcement/dqn/DeepQNetwork.py)

### Deep Learning
- DNN
- RNN
- CNN

## Examples
#### Supervised Learning

- [Logistic Regression](./examples/example_LogisticRegression.py)
- [Linear regression](./examples/example_LinearRegression.py)
, [Lasso](./examples/example_LassoRegression.py)
, [Ridge](./examples/example_RidgeRegression.py)
, [Polynomial ridge regression](./examples/example_PolynomialRidgeRegression.py)
- [Linear Discriminant Analysis](./examples/example_PCA_LDA.py)
- [k-Nearest Neighbors](./examples/example_KNeighborsClassifier.py)
- [Perceptron](./examples/example_Perceptron.py)
- [Support Vector Machine 01](./examples/example_svm.py), [02](./examples/example_svm_02.py)
- [Adaboost](./examples/example_Adaboost.py)
- [Classification tree](./examples/example_ClassificationTree.py), [Regression tree](./examples/example_RegressionTree.py)
- [GradientBoosting classifier](./examples/example_GradientBoostingClassifier.py), [GradientBoosting regressor](./examples/example_GradientBoostingRegressor.py)
- [Random Forest](./examples/example_RandomForestClassifier.py)
- [XGBoost](./examples/example_XGBoost.py)
- [Naive Bayes](./examples/example_GaussianNB.py)

#### Unsupervised Learning
- [PCA](./examples/example_PCA_LDA.py)
- [K-Means](./examples/example_KMeans.py)
- [Deep Q Network](./example_DeepQNetwork.py)

#### Reinforcement Learning
- [Q-learning](./examples/example_QLearning.py)
- [SARSA](./examples/example_SARSA.py)

## Reference
- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [Lasso](https://github.com/satopirka/Lasso)
- [Implementing a Principal Component Analysis (PCA)](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)
- [Linear Discriminant Analysis Bit by Bit](https://sebastianraschka.com/Articles/2014_python_lda.html)
- [Reinforcement Learning](https://github.com/rlcode/reinforcement-learning)