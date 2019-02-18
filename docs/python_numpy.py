import numpy as np
X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

""" shape
shape of X = 3 x 4
"""
print(X.shape[0])


""" X[:, 1:]

=> [[ 2  3  4]
    [ 6  7  8]
    [10 11 12]]
"""
print(X[:, 1:])


""" X[:, j]
=> [ 1  5  9]
   [ 2  6 10]
   ...
"""
for i in range(4):
    print(X[:, i])

""" zero
[0. 0. 0.]
"""
print(np.zeros(3))

""" np.outer()
Compute the outer product of two vectors.

[[1 2 3]
 [2 4 6]
 [3 6 9]]
"""
print(np.outer([1, 2, 3], [1, 2, 3]))
