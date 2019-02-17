# Machine learning algorithms

## Getting Started
Minimal implementation of machine learning algorithms in python. This project is for learning algorithms.

## Usage
Run exmaple_*.py under examples/ folder.

## Mathematics and Python
Read theses simple tutorials which used in 'mlalgo' before learning algorithms.
* [Python classes](./math_and_python/python_class.py)
* [Numpy](./math_and_python/python_numpy.py)
* [Singular value decomposition](./math_and_python/singular_value_decomposition.py)

## Algorithms
### Supervised Learning
### [Regression](./mlalgo/supervised/regression.py)

| | Linear Regression | Lasso Regression | Ridge Regression |
| --- | --- | --- | --- |
| Loss function | Least squared error | L1 regularization| L2 regularization |
| Algorithms | SVD | Coordinate descent | Gradient descent |
| Papers | | [Solve lasso by Coordinate Descent](https://core.ac.uk/download/pdf/6287975.pdf) | |
| Examples | [ Linear Regression](./examples/example_LinearRegression.py) | [Lasso Regression](./examples/example_LassoRegression.py)| [Ridge Regression](./examples/example_RidgeRegression.py) |

### [Support Vector Machine](./mlalgo/supervised/support_vector_machine.py)

* Algorithms: Sequential Minimal Optimization(SMO)
* Papers: [SMO: A Fast Algorithm for Training Support Vector Machines](https://pdfs.semanticscholar.org/59ee/e096b49d66f39891eb88a6c84cc89acba12d.pdf)
* Examples: [Support Vector Machine](./examples/example_svm.py)

### [Decision Trees](./mlalgo/supervised/decision_tree.py)

| | Classification Tree | RegressionTree |
| --- | --- | --- |
| Split | Information gain | Variance reduction |
| Predict | Majority vote | Mean |
| Examples | [Example - Classification Tree](./examples/example_ClassificationTree.py) | [Example - Regression Tree](./examples/example_RegressionTree.py) |

### [Linear Discriminant Analysis](./mlalgo/supervised/linear_discriminant_analysis.py)

* Algorithms: maximize the separation between multiple classes
* Examples: [Example](./examples/example_PCA_LDA.py)

---

### Unsupervised Learning
### [Principal Component Analysis](./mlalgo/unsupervised/principal_component_analysis.py)

* Algorithms: maximize the variance of our data
* Examples: [Example](./examples/example_PCA_LDA.py)

---

### Reinforcement Learning
* in progress...

---

### Deep Learning
* in progress...