import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(os.path.realpath(__file__)))


class Lasso:
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, fit_intercept: bool = True) -> None:
        self.alpha: float = alpha  # 正則化項の係数
        self.max_iter: int = max_iter  # 繰り返しの回数
        self.fit_intercept: bool = fit_intercept  # 切片(i.e., \beta_0)を用いるか
        self.coef_ = None  # 回帰係数(i.e., \beta)保存用変数
        self.intercept_ = None  # 切片保存用変数

    def _soft_thresholding_operator(self, x: float, lambda_: float) -> float:
        if x > 0.0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0.0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))

        beta = np.zeros(X.shape[1])
        if self.fit_intercept:
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        for iteration in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                tmp_beta = beta.copy()
                tmp_beta[j] = 0.0
                r_j = y - np.dot(X, tmp_beta)
                arg1 = np.dot(X[:, j], r_j)
                arg2 = self.alpha * X.shape[0]

                beta[j] = self._soft_thresholding_operator(
                    arg1, arg2) / (X[:, j]**2).sum()

                if self.fit_intercept:
                    beta[0] = np.sum(
                        y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta

        return self

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        if self.fit_intercept:
            y += self.intercept_ * np.ones(len(y))
        return y


df = pd.read_csv("Boston.csv", index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean()) / df.std()  # 基準化
X = df.iloc[:, :13].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = Lasso(alpha=0.1, max_iter=1000)

y_pred_lasso = model.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

print(model.intercept_)
print(model.coef_)
print("r^2 on test data : %f" % r2_score_lasso)
