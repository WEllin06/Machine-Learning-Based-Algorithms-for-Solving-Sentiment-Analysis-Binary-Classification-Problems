import numpy as np
from Read_Split_Data import SplitDataSet

def sigmoid(z):
    """
    Sigmoid函数的实现

    参数：
    z -- 输入向量，类型为numpy数组

    返回：
    sigmoid_output -- Sigmoid函数的输出，类型为numpy数组
    """
    sigmoid_output = 1 / (1 + np.exp(-z))
    return sigmoid_output

def cross_entropy_loss(y_true, y_pred):
    """
    交叉熵损失函数的实现

    参数：
    y_true -- 真实标签，类型为numpy数组
    y_pred -- 预测概率，类型为numpy数组

    返回：
    loss -- 交叉熵损失，标量
    """
    # 防止log(0)的情况
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 计算交叉熵损失
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def calculate_accuracy(y_true, y_pred):
    # 将预测值转换为二分类结果
    y_pred_binary = (y_pred > 0.5).astype(int)  # 将预测值大于 0.5 的置为 1，否则置为 0
    # 计算准确率
    accuracy = np.mean(y_true == y_pred_binary)  # 计算准确率
    # 计算预测正确和错误的数量
    correct_predictions = np.sum(y_true == y_pred_binary)  # 计算预测正确的数量
    incorrect_predictions = np.sum(y_true != y_pred_binary)  # 计算预测错误的数量
    return accuracy, correct_predictions, incorrect_predictions

def softmax_binary_classification_SGD():
    # 准备数据
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

    # 初始化权重
    num_features = X_train.shape[1]
    W = np.zeros(num_features)

    # 设置超参数
    learning_rate = 0.01
    num_iterations = 10000
    tolerance = 1e-4
    batch_size = 32

    # 训练模型
    for i in range(num_iterations):
        # 随机选择一个小批量样本
        indices = np.random.choice(len(y_train), batch_size, replace=False)
        X_batch, y_batch = X_train[indices], y_train[indices]

        # 计算预测概率
        scores = np.dot(X_batch, W)
        y_pred = sigmoid(scores)

        # 计算梯度
        gradient = np.dot(X_batch.T, y_pred - y_batch) / len(y_batch)

        # 更新权重
        W -= learning_rate * gradient

        # 计算损失
        loss = cross_entropy_loss(y_batch, y_pred)

        # 检查收敛
        if np.linalg.norm(gradient) < tolerance:
            break

    # 在测试集上进行预测
    scores_test = np.dot(X_test, W)
    y_pred_test = sigmoid(scores_test)

    # 计算准确率
    accuracy, correct_count, incorrect_count = calculate_accuracy(y_test, y_pred_test)

    return correct_count, incorrect_count, accuracy

# 调用softmax_binary_classification_SGD()函数来执行训练和评估，并返回模型的准确率
correct_count, incorrect_count, accuracy = softmax_binary_classification_SGD()

print("测试集中分类正确的样本数量:", correct_count)
print("测试集中分类错误的样本数量:", incorrect_count)
print("模型准确率:", accuracy)
