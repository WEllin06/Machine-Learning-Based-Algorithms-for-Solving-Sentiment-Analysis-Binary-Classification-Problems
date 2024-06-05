import numpy as np
from Read_Split_Data import SplitDataSet
from sklearn.linear_model import LinearRegression

def calculate_accuracy(y_true, y_pred):
    # 将预测值转换为二分类结果
    y_pred_binary = (y_pred > 0.5).astype(int)  # 将预测值大于 0.5 的置为 1，否则置为 0
    # 计算准确率
    accuracy = np.mean(y_true == y_pred_binary)  # 计算准确率
    # 计算预测正确和错误的数量
    correct_predictions = np.sum(y_true == y_pred_binary)  # 计算预测正确的数量
    incorrect_predictions = np.sum(y_true != y_pred_binary)  # 计算预测错误的数量
    return accuracy, correct_predictions, incorrect_predictions

def linregression():
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

    # 创建线性回归模型
    model = LinearRegression()

    # 拟合模型
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy, correct_predictions, incorrect_predictions = calculate_accuracy(y_test, y_pred)

    return accuracy, correct_predictions, incorrect_predictions

# 调用 linregression() 函数来执行训练和评估，并返回模型准确率、正确预测数据、错误预测数据
accuracy, correct_predictions, incorrect_predictions = linregression()

print("Accuracy:", accuracy)
print("Correct Predictions:", correct_predictions)
print("Incorrect Predictions:", incorrect_predictions)
