from Read_Split_Data import SplitDataSet
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

def softmax_regression():
    # 准备数据
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

    # 对标签进行独热编码
    encoder = OneHotEncoder()
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

    # 构建 Softmax 回归模型
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    # 训练模型
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = model.score(X_test, y_test)

    # 计算正确分类样本总数量和错误分类样本总数量
    correct_count = sum(y_pred == y_test)
    error_count = len(y_test) - correct_count

    return accuracy, correct_count, error_count

# 调用函数并获取结果
accuracy, correct_count, error_count = softmax_regression()
print("Accuracy:", accuracy)
print("Correct Count:", correct_count)
print("Error Count:", error_count)
