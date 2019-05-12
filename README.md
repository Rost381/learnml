# learnml
![](https://img.shields.io/badge/python-3.6+-blue.svg)

Write machine learning algorithms from zero for self-learning.

All algorithm codes are implemented in Python with friendly comments and easier to understand how they works.

## Documentation
Friendly documentation is available at [https://byzhi.github.io/learnml/](https://byzhi.github.io/learnml/). 

## Algorithms

### Supervised Learning
Linear models

| Linear models | examples | docs |
| --- | --- | --- |
| [Logistic Regression](./learnml/supervised/logistic_regression.py) | [example](./examples/example_LogisticRegression.py) | [docs](https://byzhi.github.io/learnml/supervised/logistic_regression.html) |
| [Regression](./learnml/supervised/regression.py) | [Linear](./examples/example_LinearRegression.py), [Lasso](./examples/example_LassoRegression.py), [Ridge](./examples/example_RidgeRegression.py), [Polynomial ridge](./examples/example_PolynomialRidgeRegression.py) | [docs](https://byzhi.github.io/learnml/supervised/linear_regression.html) |

Classification

- Linear Discriminant Analysis | [code](./learnml/supervised/linear_discriminant_analysis.py), [example](./examples/example_PCA_LDA.py)
- k-Nearest Neighbors | [code](./learnml/supervised/k_nearest_neighbors.py), [example](./examples/example_KNeighborsClassifier.py)
- Perceptron | [code](./learnml/supervised/perceptron.py), [example](./examples/example_Perceptron.py)
- Support Vector Machine | [code](./learnml/supervised/support_vector_machine.py), examples: [01](./examples/example_svm.py), [02](./examples/example_svm_02.py)

Tree-based and ensemble methods

- Adaboost | [code](./learnml/supervised/adaboost.py), [example](./examples/example_Adaboost.py)
- Decision Tree | [code](./learnml/supervised/decision_tree.py), examples: [Classification tree](./examples/example_ClassificationTree.py), [Regression tree](./examples/example_RegressionTree.py)
- Gradient boosting | [code](./learnml/supervised/gradient_boosting.py), examples: [GradientBoosting classifier](./examples/example_GradientBoostingClassifier.py), [GradientBoosting regressor](./examples/example_GradientBoostingRegressor.py)
- Random forests | [code](./learnml/supervised/random_forest.py), [example](./examples/example_RandomForestClassifier.py)
- XGBoost | [code](./learnml/supervised/xgboost.py), [example](./examples/example_XGBoost.py)

Generative Learning

- Naive Bayes | [code](./learnml/supervised/naive_bayes.py), [example](./examples/example_GaussianNB.py)

### Unsupervised Learning

Dimension reduction

- K-Means | [code](./learnml/unsupervised/kmeans.py), [example](./examples/example_KMeans.py)
- Principal Component Analysis | [code](./learnml/unsupervised/principal_component_analysis.py), [example](./examples/example_PCA_LDA.py)


### Reinforcement Learning
- Q-learning | [code](./learnml/reinforcement/qlsarsa/base.py), [example](./examples/example_QLearning.py)
- SARSA | [code](./learnml/reinforcement/qlsarsa/base.py), [example](./examples/example_SARSA.py)
- DQN (Deep Q Network) | [code](./learnml/reinforcement/dqn/DeepQNetwork.py), [example](./examples/example_DeepQNetwork.py)

### Deep Learning
- DNN
- RNN
- CNN [example](./examples/example_CNN.py)

## Installation
```
git clone https://github.com/byzhi/learnml
cd learnml
pip install -e .
```

## Usage
```python
from learnml.api import LinearRegression
model = LinearRegression()
model.fit(X, y)
```

## Reference
- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [Lasso](https://github.com/satopirka/Lasso)
- [Implementing a Principal Component Analysis (PCA)](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)
- [Linear Discriminant Analysis Bit by Bit](https://sebastianraschka.com/Articles/2014_python_lda.html)
- [Reinforcement Learning](https://github.com/rlcode/reinforcement-learning)