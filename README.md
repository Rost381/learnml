# Machine learning algorithms For Student

## Getting Started
Implementation of Machine Learning Algorithms Step by Step As Student

## Usage
Run exmaple_*.py under examples/ folder

## Algorithms
### Getting started
* [Simple Linear Regression](./ml_student/simple_linear_regression/simple_linear_regression.py) | [Example](./examples/example_SimpleLinearRegression.py)

### Supervised Learning
* [Linear Regression](./ml_student/regression/regression.py)

| Option | Description |
| ------ | ----------- |
| Loss function | Least squared error |
| Core | X = U * Sigma * Adjugate(V)<br>X+ = V * pseudo-inverse(Sigma) * Adjugate(U)<br>w = V * pseudo-inverse(Sigma) * Adjugate(U) * X * y |
| SVD | ![](images/svd.png) |
| Example | [One-dimensional linear regression](./examples/example_LinearRegression.py) |
| Example | [Multiple linear regression]() |

* [Logistic Regression](./ml_student/logistic_regression/logistic_regression.py)
  * [Example](./examples/example_logit.py)
* [k Nearest Neighbors](./ml_student/k_nearest_neighbors/k_nearest_neighbors.py)
  * [Example](./examples/example_knn.py)
* [Principal Component Analysis](./ml_student/linear_discriminant_analysis/linear_discriminant_analysis.py)
  * [Example](./examples/example_lda.py)
* [Support Vector Machine](./ml_student/support_vector_machine/support_vector_machine.py)
  * [Example](./examples/example_svm.py)
* [Decision Trees](./ml_student/decision_tree/decision_tree.py)
  * [Example - Classification Tree](./examples/example_ct.py)
  * [Example - Regression Tree](./examples/example_rt.py)

### Unsupervised Learning
* [Linear Discriminant Analysis](./ml_student/principal_component_analysis/principal_component_analysis.py)
  * [Example](./examples/example_pca.py)

### Reinforcement Learning
* in progress...

### Deep Learning
* [Back Propagation](./ml_student/back_propagation/back_propagation.py)
  * [Example](./examples/example_bp.py)