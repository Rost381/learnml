# Machine learning algorithms

## Getting Started
Minimal implementation of machine learning algorithms from Zero. This project is for learning algorithms.

## Usage
Run exmaple_*.py under examples/ folder.

## Mathematics and Python
Read theses simple tutorials which used in 'mlalgo' before learning algorithms.
* Mathematics
  * [Singular value decomposition(SVD)](./docs/singular_value_decomposition.py)
  * [Quadratic program](./docs/quadratic_program.py)
* Python
  * [Classes](./docs/python_class.py)
  * [Numpy](./docs/python_numpy.py)

## Algorithms
### Supervised Learning
* #### Regression

| Linear Regression | Lasso Regression | Ridge Regression |
| --- | --- | --- |
| Least squared error | L1 regularization| L2 regularization |
| SVD [[Code](./mlalgo/supervised/regression.py)] | Coordinate descent [[Paper](https://core.ac.uk/download/pdf/6287975.pdf)] [[Code](./mlalgo/supervised/regression.py)] | Gradient descent [[Code](./mlalgo/supervised/regression.py)] |
| [Example](./examples/example_LinearRegression.py) | [Example](./examples/example_LassoRegression.py)| [Example](./examples/example_RidgeRegression.py) |

* #### Support Vector Machine
  * Convex optimization
  [[Code](./mlalgo/supervised/support_vector_machine_cvxopt.py)]
  [[Examples](./examples/example_svmCVXOPT.py)]


  * SMO
  [[Papers](https://pdfs.semanticscholar.org/59ee/e096b49d66f39891eb88a6c84cc89acba12d.pdf)]
  [[Code](./mlalgo/supervised/support_vector_machine_smo.py)]
  [[Examples](./examples/example_svmSMO.py)]

* #### Decision Trees

| Classification Tree | RegressionTree |
| --- | --- |
| Information gain / Majority vote| Variance reduction / Mean |
| [Code](./mlalgo/supervised/decision_tree.py) | [Code](./mlalgo/supervised/decision_tree.py) |
[Example](./examples/example_ClassificationTree.py) | [Example](./examples/example_RegressionTree.py) |

* #### Linear Discriminant Analysis
    * Maximize the separation between multiple classes. [[Code](./mlalgo/supervised/linear_discriminant_analysis.py)] [[Example](./examples/example_PCA_LDA.py)]

---

### Unsupervised Learning
* #### Principal Component Analysis
  * Maximize the variance of our data. [[Code](./mlalgo/unsupervised/principal_component_analysis.py)]  [[Example](./examples/example_PCA_LDA.py)]

---

### Reinforcement Learning
* in progress...

---

### Deep Learning
* in progress...