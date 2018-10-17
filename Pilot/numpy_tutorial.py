import numpy as np
x = np.zeros(4, dtype=int)

print(x)

# argmin
def selection_sort(x):
    for i in range(len(x)):
        # numpy.argmin
        # Returns the indices of the minimum values along an axis.
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x


x = np.array([2, 1, 4, 3, 5])
print(selection_sort(x))

# reshape
x = np.arange(16).reshape(4, 4)
print(x[0:3, 0:2])
