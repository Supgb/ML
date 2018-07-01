"""
author: Supgb
Title: Polynomial Regression
Last modified： Jun. 30, 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# plt.plot(X, y, '.')
# plt.show()
#
# X_poly = np.c_[X, X**2]
#
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y)
# print(lin_reg.intercept_, lin_reg.coef_)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
intercept = lin_reg.intercept_
coef = lin_reg.coef_


X_new = np.arange(-2, 2, 0.01)
X_new = X_new.reshape(-1, 1)
X_new_poly = poly_features.fit_transform(X_new)
y_pred = lin_reg.predict(X_new_poly)
y_target = 0.5*X_new**2 + X_new + 2
plt.plot(X, y, '.', X_new, y_pred, 'r-', X_new, y_target, 'g--')
plt.show()