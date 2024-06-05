import numpy as np
from Read_Split_Data import SplitDataSet  # 导入数据集分割的函数


# 计算岭回归的损失函数
def ridge_regression_loss(X, y, w, alpha):
    n_samples = X.shape[0]
    error = y - np.dot(X, w)
    loss = np.mean(error ** 2) + alpha * np.sum(w ** 2)
    return loss


# 使用岭回归的闭式解求解参数 w
def ridge_regression_closed_form(X, y, alpha):
    # 添加偏置项
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # 在特征矩阵 X 前添加一列全为 1 的偏置项,提高模型的泛化能力
    n_features = X_with_bias.shape[1]

    # 计算闭式解
    XtX = np.dot(X_with_bias.T, X_with_bias)
    w = np.dot(np.linalg.inv(XtX + alpha * np.eye(n_features)), np.dot(X_with_bias.T, y))

    return w


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

# 使用岭回归的闭式解求解参数 w
alpha = 0.001  # 正则化参数
weights = ridge_regression_closed_form(X_train, y_train, alpha=alpha)

# 在测试集上进行预测
X_test_with_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])  # 添加偏置项
y_pred_test = np.dot(X_test_with_bias, weights)  # 使用训练好的模型在测试集上进行预测

# 计算准确率、预测正确和错误的数量
accuracy, correct_predictions, incorrect_predictions = calculate_accuracy(y_test, y_pred_test)  # 计算预测的准确率、正确和错误的预测数量

print("Correct predictions:", correct_predictions)  # 打印正确的预测数量
print("Incorrect predictions:", incorrect_predictions)  # 打印错误的预测数量
print("Accuracy:", accuracy)  # 打印准确率
