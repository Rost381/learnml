import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(os.path.realpath(__file__)))


df = pd.read_csv("Boston.csv", index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean())/df.std() # 基準化
X = df.iloc[:, :13].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = Lasso(alpha=0.1)

y_pred_lasso = model.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

print(model.intercept_)
print(model.coef_)
print("r^2 on test data : %f" % r2_score_lasso)

X = [[1,2,3],[4,5,6],[7,8,9]]
print(X)
X1 = np.column_stack((np.ones(len(X)), X))
X2 = np.insert(X, 0, 1, axis=1)
beta = np.zeros(X1.shape[1])
print(X1.shape)
print(beta)

w = [[1,2],[3,4]]
print(np.linalg.norm(w))

x = 1
l = []
l.append(x)
print(l)