import matplotlib.pyplot as plt
import numpy as np


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100,1)), X] # 加入theta0项

n_epochs = 50 # 训练次数为50次
eta = 0.1 # 学习率
m = 100 # 数据集大小，即实例个数
theta = np.random.randn(2, 1)
# 随机生成theta值

#开始训练

for epoch in range(n_epochs):
    for i in range(m): # 注意 m 为数据集的大小
        random_index = np.random.randint(m) # 随机下标的生成
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        # 梯度向量的计算
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        # 更新 theta 值
        theta = theta - eta * gradients

X_new = np.array([[0], [2]]) # 生成新数据
X_new_b = np.c_[np.ones((2, 1)), X_new] # 加入theta0项
y_pred = X_new_b.dot(theta) # 预测
print(theta)
plt.plot(X, y, '.', X_new, y_pred)
plt.show()