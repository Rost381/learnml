# Principal component analysis

Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.

## Solution

### Step 1

Caculate eigenvalues and eigenvectors of covariance_matrix(X)

```python
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```

### Step 2

Sort eigenvectors from largest to smallest

```python
idx = eigenvalues.argsort()[::-1]
```

### Step 3

Select the first n_components of eigenvalues

```python
eigenvalues = eigenvalues[idx][:self.n_components]
eigenvectors = eigenvectors[:, idx][:, :self.n_components]
```

### Predict

```python
X_transformed = X.dot(eigenvectors)
```