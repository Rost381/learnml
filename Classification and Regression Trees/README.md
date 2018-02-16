# Classification and Regression Trees

> Decision tree learning uses a predictive model to go from observations about an item to conclusions about the item's target value.

## Classification trees

Tree models where the target variable can take a discrete set of values are called classification trees

## Decision trees

Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.

There are many specific decision-tree algorithms, like:

* ID3 (Iterative Dichotomiser 3)
* C4.5 (successor of ID3)
* CART (Classification And Regression Tree)

## Gini index

The Gini index is the name of the cost function used to evaluate splits in the dataset.

$$
Gini(p) = \sum\limits_{k = 1}^K {p{}_k(1 - {p_k})} = 1 - \sum\limits_{k = 1}^K {p_k^2}
$$

## Information gain

Information gain is based on the concept of entropy.

$$
{\displaystyle H(T)=I_{E}(p_{1},p_{2},...,p_{J})=-\sum _{i=1}^{J}p_{i}\log _{2}^{}p_{i}}
$$