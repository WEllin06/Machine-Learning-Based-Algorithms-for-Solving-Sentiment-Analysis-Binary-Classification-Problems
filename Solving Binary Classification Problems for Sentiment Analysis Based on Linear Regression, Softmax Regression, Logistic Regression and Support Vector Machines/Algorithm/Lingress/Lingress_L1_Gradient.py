import numpy as np
from Read_Split_Data import SplitDataSet  # 导入数据集分割的函数

# 实现 L1 正则化（Lasso 回归）模型的参数更新
def lasso_regression_predict(X, y, alpha, learning_rate, epochs):
    # 添加偏置项
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # 在特征矩阵 X 前添加一列全为 1 的偏置项

    # 初始化参数向量
    n_samples, n_features = X_with_bias.shape
    weights = np.zeros(n_features)

    # 使用梯度下降求解 Lasso 回归模型参数
    for _ in range(epochs):
        # 计算预测值
        y_pred = np.dot(X_with_bias, weights)
        # 计算平方误差
        residuals = y_pred - y
        # 计算梯度
        gradient = np.dot(X_with_bias.T, residuals) / n_samples
        # 更新权重参数
        weights -= learning_rate * (gradient + alpha * np.sign(weights))

    return weights

# 计算准确率、预测正确和错误的数量
def calculate_accuracy(y_true, y_pred):
    # 将预测值转换为二分类结果
    y_pred_binary = (y_pred > 0.5).astype(int)  # 将预测值大于 0.5 的置为 1，否则置为 0
    # 计算准确率
    accuracy = np.mean(y_true == y_pred_binary)  # 计算准确率
    # 计算预测正确和错误的数量
    correct_predictions = np.sum(y_true == y_pred_binary)  # 计算预测正确的数量
    incorrect_predictions = np.sum(y_true != y_pred_binary)  # 计算预测错误的数量
    return accuracy, correct_predictions, incorrect_predictions

# 读取数据集并进行训练集和测试集的分割
X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()  # 使用分割数据集的函数得到训练集和测试集

# 使用 L1 正则化（Lasso 回归）求解线性回归模型的参数
alpha = 0.01  # 正则化参数
learning_rate = 0.01  # 学习率
epochs = 4000  # 迭代次数
weights = lasso_regression_predict(X_train, y_train, alpha, learning_rate, epochs)

# 在测试集上进行预测
X_test_with_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])  # 添加偏置项
y_pred_test = np.dot(X_test_with_bias, weights)  # 使用训练好的模型在测试集上进行预测

# 计算准确率、预测正确和错误的数量
accuracy, correct_predictions, incorrect_predictions = calculate_accuracy(y_test, y_pred_test)  # 计算预测的准确率、正确和错误的预测数量

print("Correct predictions:", correct_predictions)  # 打印正确的预测数量
print("Incorrect predictions:", incorrect_predictions)  # 打印错误的预测数量
print("Accuracy:", accuracy)  # 打印准确率
