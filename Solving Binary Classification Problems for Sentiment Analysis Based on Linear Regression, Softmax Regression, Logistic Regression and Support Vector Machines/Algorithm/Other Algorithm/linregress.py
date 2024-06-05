from Read_Split_Data import SplitDataSet

import numpy as np


def simple_linear_regression(X_train, y_train, X_test, y_test):
    # 选择第一个特征作为自变量
    x_train = X_train[:, 0]

    # 计算自变量和因变量的均值
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)

    # 计算 w 的分子和分母
    numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
    denominator = np.sum((x_train - x_mean) ** 2)

    # 计算 w
    w = numerator / denominator

    # 计算 b
    b = y_mean - w * x_mean

    # 在测试集上进行预测
    x_test = X_test[:, 0]
    y_pred = w * x_test + b

    # 计算 R 平方
    total_sum_squares = np.sum((y_test - np.mean(y_test)) ** 2)
    residual_sum_squares = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - (residual_sum_squares / total_sum_squares)

    return r2

# 训练集和测试集的划分
X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

# 计算简单线性回归的参数并返回 R 平方
def linregression():
    accuracy = simple_linear_regression(X_train, y_train, X_test, y_test)
    return accuracy

# 调用 linregression() 函数来执行训练和评估，并返回模型准确率
accuracy = linregression()

'''
def ridge_regression(X_train, y_train, X_test, y_test, alpha=0.1):
    # 添加一个偏置项（常数项）到 X_train 和 X_test
    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    X_test = np.column_stack((np.ones(len(X_test)), X_test))

    # 计算回归系数
    n_features = X_train.shape[1]
    A = X_train.T @ X_train + alpha * np.eye(n_features)
    w = np.linalg.inv(A) @ X_train.T @ y_train

    # 在测试集上进行预测
    y_pred = X_test @ w

    # 计算 R 平方
    total_sum_squares = np.sum((y_test - np.mean(y_test)) ** 2)
    residual_sum_squares = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - (residual_sum_squares / total_sum_squares)

    return r2

# 训练集和测试集的划分
X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

# 计算岭回归的参数并返回 R 平方
def linregression():
    accuracy = ridge_regression(X_train, y_train, X_test, y_test)
    return accuracy

# 调用 linregression() 函数来执行训练和评估，并返回模型准确率
accuracy = linregression()

'''

'''
import SplitDataSet
from sklearn.linear_model import LinearRegression


def linregression():
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

    # 创建线性回归模型
    model = LinearRegression()

    # 拟合模型
    model.fit(X_train, y_train)

    # 在测试集上评估模型
    accuracy = model.score(X_test, y_test)

    return accuracy

# 调用 linregression() 函数来执行训练和评估，并返回模型准确率
accuracy = linregression()
'''
