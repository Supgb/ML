"""
author: Supgb
title: Logistic Regression
last modified: Jun.30, 2018
"""

from sklearn import datasets
iris = datasets.load_iris()

import numpy as np
X = iris['data'][:, 3:]
y = (iris['target']==2).astype(np.int)

from sklearn.linear_model import LogisticRegression
Log_reg = LogisticRegression()
Log_reg.fit(X, y)

import matplotlib.pyplot as plt
X_new = np.arange(0, 3, 0.01).reshape(-1, 1)
y_prob = Log_reg.predict_proba(X_new)
# 可视化处理
plt.plot(X_new, y_prob[:, 0], 'b--', label='Non-Virginica')
plt.plot(X_new, y_prob[:, 1], 'r-', label='Virginica')
# 显示决策边界
x = np.arange(0, 3, 0.01)
idx = np.argwhere(np.diff(np.sign(y_prob[:, 0] - y_prob[:, 1])) != 0).reshape(-1) + 0
plt.plot(x[idx[0]], y_prob[:, 0][idx[0]], 'ro')
plt.axvline(x[idx[0]],ls=':', label='Decision boundary')
plt.xlabel('petal width')
plt.ylabel('probability')
plt.legend()
plt.show()