# Machine learning algorithms

## Getting Started
Minimal implementation of machine learning algorithms in python. This project is for learning algorithms.

## Usage
Run exmaple_*.py under examples/ folder.

## Mathematics and Python
Read theses simple tutorials which used in 'mlalgo' before learning algorithms.
* Mathematics
  * [Singular value decomposition(SVD)](./docs/singular_value_decomposition.py)
* Python
  * [Classes](./docs/python_class.py)
  * [Numpy](./docs/python_numpy.py)

## Algorithms
### Supervised Learning
* #### [Regression](./mlalgo/supervised/regression.py)

| | Linear Regression | Lasso Regression | Ridge Regression |
| --- | --- | --- | --- |
| Loss function | Least squared error | L1 regularization| L2 regularization |
| Algorithms | SVD | Coordinate descent | Gradient descent |
| Papers | | [Solve lasso by Coordinate Descent](https://core.ac.uk/download/pdf/6287975.pdf) | |
| Examples | [ Linear Regression](./examples/example_LinearRegression.py) | [Lasso Regression](./examples/example_LassoRegression.py)| [Ridge Regression](./examples/example_RidgeRegression.py) |

* #### Support Vector Machine
  * Algorithms: Convex optimization
  [[Code](./mlalgo/supervised/support_vector_machine_cvxopt.py)]
  [[Examples](./examples/example_svmCVXOPT.py)]


  * Algorithms: SMO
  [[Papers](https://pdfs.semanticscholar.org/59ee/e096b49d66f39891eb88a6c84cc89acba12d.pdf)]
  [[Code](./mlalgo/supervised/support_vector_machine_smo.py)]
  [[Examples](./examples/example_svmSMO.py)]

* #### [Decision Trees](./mlalgo/supervised/decision_tree.py)

| | Classification Tree | RegressionTree |
| --- | --- | --- |
| Split | Information gain | Variance reduction |
| Predict | Majority vote | Mean |
| Examples | [Example - Classification Tree](./examples/example_ClassificationTree.py) | [Example - Regression Tree](./examples/example_RegressionTree.py) |

* #### [Linear Discriminant Analysis](./mlalgo/supervised/linear_discriminant_analysis.py)

  * Algorithms: maximize the separation between multiple classes
  * Examples: [Example](./examples/example_PCA_LDA.py)

---

### Unsupervised Learning
* #### [Principal Component Analysis](./mlalgo/unsupervised/principal_component_analysis.py)

  * Algorithms: maximize the variance of our data
  * Examples: [Example](./examples/example_PCA_LDA.py)

---

### Reinforcement Learning
* in progress...

---

### Deep Learning
* in progress...