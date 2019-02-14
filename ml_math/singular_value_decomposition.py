import numpy as np

"""
A = U . s . V^T
A+ = V . S^T . U^T

s = [[s11, 0,   0],
     [0, s22,   0],
     [0,   0, s33]]

S = [[1 / s11, 0,       0],
     [0,       1 / s22, 0],
     [0,       0,       1 / s33]]

np.linalg.pinv(A) = V_T.T.dot(S.T).dot(U.T)

https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
"""
A = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]])
U, s, V_T = np.linalg.svd(A)

d = 1.0 / s
# m x n S matrix
S = np.zeros(A.shape)
# n x n diagonal matrix
S[:A.shape[1], :A.shape[1]] = np.diag(d)
# calculate pseudoinverse
B = V_T.T.dot(S.T).dot(U.T)
print(B)


A = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]])
# calculate pseudoinverse
B = np.linalg.pinv(A)
print(B)
