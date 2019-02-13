# Machine learning algorithms For Student

## Getting Started
Implementation of Machine Learning Algorithms Step by Step As Student

## Usage
Run exmaple_*.py under examples/ folder

## Algorithms
### Getting started
* [Simple Linear Regression](./ml/simple_linear_regression/simple_linear_regression.py) | [Example](./examples/example_SimpleLinearRegression.py)

### Supervised Learning
* [Linear Regression](./ml/regression/regression.py)

| Option | Description |
| ------ | ----------- |
| Loss function | Least squared error |
| Core | X+ = (X_t * X)^-1 * X_t<br>X = U * Sigma * V<br>X+ = V * pseudo-inverse(Sigma) * Adjugate(U)<br>w = U * pseudo-inverse(Sigma) * Adjugate(U) * X * y |
| SVD | ![](images/svd.png) |
| Example | [One-dimensional linear regression](./examples/example_LinearRegression.py) |
| Example | [Multiple linear regression]() |

* [Logistic Regression](./ml/logistic_regression/logistic_regression.py)
  * [Example](./examples/example_logit.py)
* [k Nearest Neighbors](./ml/k_nearest_neighbors/k_nearest_neighbors.py)
  * [Example](./examples/example_knn.py)
* [Principal Component Analysis](./ml/linear_discriminant_analysis/linear_discriminant_analysis.py)
  * [Example](./examples/example_lda.py)
* [Support Vector Machine](./ml/support_vector_machine/support_vector_machine.py)
  * [Example](./examples/example_svm.py)
* [Decision Trees](./ml/decision_tree/decision_tree.py)
  * [Example - Classification Tree](./examples/example_ct.py)
  * [Example - Regression Tree](./examples/example_rt.py)

### Unsupervised Learning
* [Linear Discriminant Analysis](./ml/principal_component_analysis/principal_component_analysis.py)
  * [Example](./examples/example_pca.py)

### Reinforcement Learning
* in progress...

### Deep Learning
* [Back Propagation](./ml/back_propagation/back_propagation.py)
  * [Example](./examples/example_bp.py)