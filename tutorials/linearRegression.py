import numpy as np
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100, 1)
y = 3 * X + 4 + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

eta = 0.01 # 学习率
n_iterations = 1000 # 迭代次数
m = 100 # 因为我们的X有100行对应的就是100个实例

theta = np.random.randn(2,1) # 随机生成权重

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - eta * gradients

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred = X_new_b.dot(theta)
plt.plot(X, y, '.', X_new, y_pred)
plt.show()