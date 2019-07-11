import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from learnml.supervised.api import ElasticNet


def main():
    # #############################################################################
    # Generate some sparse data to play with
    np.random.seed(42)

    n_samples, n_features = 50, 100
    X = np.random.randn(n_samples, n_features)

    # Decreasing coef w. alternated signs for visualization
    idx = np.arange(n_features)
    coef = (-1) ** idx * np.exp(-idx / 10)
    coef[10:] = 0  # sparsify coef
    y = np.dot(X, coef)

    # Add noise
    y += 0.01 * np.random.normal(size=n_samples)

    # Split data in train set and test set
    n_samples = X.shape[0]
    X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
    X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

    # #############################################################################
    # ElasticNet
    alpha = 0.1
    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

    enet.fit(X_train, y_train)
    y_pred_enet = enet.predict(X_test)
    r2_score_enet = r2_score(y_test, y_pred_enet)

    print("r^2 on test data : %f" % r2_score_enet)


if __name__ == "__main__":
    main()
