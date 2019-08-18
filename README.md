# Learnml

Python Machine Learning Algorithms Guide for Humans. ![](https://img.shields.io/badge/Python-3.6+-blue.svg)

> Someone want to learn Machine Learning based on Algorithms with Simple explanation, Mathematics Knowledge and Python code.

## Features

![](https://img.shields.io/badge/-Simple_Explanation-red.svg) ![](https://img.shields.io/badge/-Mathematics-green.svg) ![](https://img.shields.io/badge/-Python-blue.svg)
- Simple Explanation on Statistics, Machine learning including deep learning
- Implementing algorithms from scratch on Python
- Friendly docs with Simple Explanation and Python Code

## Documentation

[https://byzhi.github.io/learnml](https://byzhi.github.io/learnml/)

## Implementations

- Math
  - Notation - [doc](https://byzhi.github.io/learnml/math/notation.html)
  - Linear algebra - [doc](https://byzhi.github.io/learnml/math/linear_algebra.html)
  - Calculus
  - Probability and Distributions
  - Optimization
- Statistics
  - [Variance](https://github.com/byzhi/learnml/blob/master/learnml/utils/stats.py#L16) - [doc](https://byzhi.github.io/learnml/statistics/variance.html)
  - [Covariance matrix](https://github.com/byzhi/learnml/blob/master/learnml/utils/stats.py#L6) - [doc](https://byzhi.github.io/learnml/statistics/covariance_matrix.html)
  - [Standardize](https://github.com/byzhi/learnml/blob/master/learnml/utils/stats.py#L24) - [doc](https://byzhi.github.io/learnml/statistics/standardize.html)
  - [Normalize](https://github.com/byzhi/learnml/blob/master/learnml/utils/stats.py#L35) - [doc](https://byzhi.github.io/learnml/statistics/normalize.html)
- Supervised Learning
  - Linear models
    - [Logistic Regression](./learnml/supervised/logistic_regression.py) - [example](./examples/example_LogisticRegression.py) - [doc](https://byzhi.github.io/learnml/supervised/logistic_regression.html)
    - [Regression](./learnml/supervised/regression.py)
      - [Linear](./examples/example_LinearRegression.py) - [doc](https://byzhi.github.io/learnml/supervised/linear_regression.html)
      - [Lasso](./examples/example_LassoRegression.py)
      - [Ridge](./examples/example_RidgeRegression.py) -
      - [Polynomial ridge](./examples/example_PolynomialRidgeRegression.py)
      - [Elastic Net](./examples/example_ElasticNet.py) - [doc](https://byzhi.github.io/learnml/supervised/elastic_net.html)
  - Classification
    - [Linear Discriminant Analysis](./learnml/supervised/linear_discriminant_analysis.py) - [example](./examples/example_PCA_LDA.py) - [doc](https://byzhi.github.io/learnml/supervised/linear_discriminant_analysis.html)
    - [k-Nearest Neighbors](./learnml/supervised/k_nearest_neighbors.py) - [example](./examples/example_KNeighborsClassifier.py)
    - [Perceptron](./learnml/supervised/perceptron.py) - [example](./examples/example_Perceptron.py)
    - [Support Vector Machine](./learnml/supervised/support_vector_machine.py) - examples: [01](./examples/example_svm.py) - [02](./examples/example_svm_02.py) - [doc](https://byzhi.github.io/learnml/supervised/support_vector_machine.html)
  - Tree-based and ensemble methods
    - [Adaboost](./learnml/supervised/adaboost.py) - [example](./examples/example_Adaboost.py)
    - [Decision Tree](./learnml/supervised/decision_tree.py) - examples: [Classification](./examples/example_ClassificationTree.py), [Regression](./examples/example_RegressionTree.py)
    - [Gradient boosting](./learnml/supervised/gradient_boosting.py) - examples: [Classifier](./examples/example_GradientBoostingClassifier.py), [Regressor](./examples/example_GradientBoostingRegressor.py)
    - [Random forests](./learnml/supervised/random_forest.py) - [example](./examples/example_RandomForestClassifier.py)
    - [XGBoost](./learnml/supervised/xgboost.py) - [example](./examples/example_XGBoost.py)
  - Generative Learning
    - [Naive Bayes](./learnml/supervised/naive_bayes.py) - [example](./examples/example_GaussianNB.py)
- Unsupervised Learning
  - Dimension reduction
    - [K-Means](./learnml/unsupervised/kmeans.py) - [example](./examples/example_KMeans.py)
    - [Principal Component Analysis](./learnml/unsupervised/principal_component_analysis.py) - [example](./examples/example_PCA_LDA.py)
- Reinforcement Learning
  - On-policy
    - [SARSA](./learnml/reinforcement/qlearning_sarsa_base.py) - [example](./examples/example_SARSA.py)
  - Off-policy
    - [Monte Carlo](./learnml/reinforcement/montecarlo.py) - [example](./examples/example_MonteCarlo.py) - doc
    - [Q-learning](./learnml/reinforcement/qlearning_sarsa_base.py) - [example](./examples/example_QLearning.py) - [doc](https://byzhi.github.io/learnml/reinforcement/q_learning.html)
    - [DQN (Deep Q Network)](./learnml/reinforcement/deepqnetwork.py) - [example](./examples/example_DeepQNetwork.py)
- Deep Learning
  - CNN (Convolutional neural network)
    - [example](./examples/example_CNN.py)
    - TextCNN - Binary Sentiment Classification
  - RNN (Recurrent neural network)
    - TextRNN - Predict Next Step 
  - [LSTM](./learnml/deep/lstm.py)
    - [example](./examples/example_Lstm.py)
    - TextLSTM - Autocomplete


## Installation

```bash
git clone https://github.com/byzhi/learnml
cd learnml
pip install -e .
```

## Usage

```python
from learnml.api import LinearRegression

model = LinearRegression()
model.fit(X - y)
```

## Reference

- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [Lasso](https://github.com/satopirka/Lasso)
- [Reinforcement Learning](https://github.com/rlcode/reinforcement-learning)
- [lstm](https://github.com/nicodjimenez/lstm)
- [Implementing a Principal Component Analysis (PCA)](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)
- [Linear Discriminant Analysis Bit by Bit](https://sebastianraschka.com/Articles/2014_python_lda.html)