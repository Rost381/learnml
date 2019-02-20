# Machine learning algorithms

## Getting Started
Minimal implementation of machine learning algorithms from Zero. This project is for learning algorithms.

## Installation
```
$ git clone https://github.com/byzhi/zero
$ cd zero
$ python setup.py install
```

## Usage
Run exmaple_*.py under examples/ folder.

## Mathematics and Python
Read theses simple code used in 'Zero' before learning algorithms.
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
| SVD [[Code](./zero/supervised/regression.py)] [[Example](./examples/example_LinearRegression.py)] | Coordinate descent [[Paper](https://core.ac.uk/download/pdf/6287975.pdf)] [[Code](./zero/supervised/regression.py)] [[Example](./examples/example_LassoRegression.py)] | Gradient descent [[Code](./zero/supervised/regression.py)] [[Example](./examples/example_RidgeRegression.py)] |

* #### Support Vector Machine
    Convex optimization [[Code](./zero/supervised/support_vector_machine_cvxopt.py)] [[Example](./examples/example_svmCVXOPT.py)]
    
    SMO [[Paper](https://pdfs.semanticscholar.org/59ee/e096b49d66f39891eb88a6c84cc89acba12d.pdf)] [[Code](./zero/supervised/support_vector_machine_smo.py)] [[Example](./examples/example_svmSMO.py)]

* #### Decision Trees

| Classification Tree | RegressionTree |
| --- | --- |
| Information gain / Majority vote| Variance reduction / Mean |
| [[Code](./zero/supervised/decision_tree.py)] [[Example](./examples/example_ClassificationTree.py)] | [[Code](./zero/supervised/decision_tree.py)] [[Example](./examples/example_RegressionTree.py)] |

* #### Linear Discriminant Analysis
  Maximize the separation between multiple classes. [[Code](./zero/supervised/linear_discriminant_analysis.py)] [[Example](./examples/example_PCA_LDA.py)]

---

### Unsupervised Learning
* #### Principal Component Analysis
  Maximize the variance of our data. [[Code](./zero/unsupervised/principal_component_analysis.py)]  [[Example](./examples/example_PCA_LDA.py)]

---

### Reinforcement Learning
* in progress...

---

### Deep Learning
* in progress...