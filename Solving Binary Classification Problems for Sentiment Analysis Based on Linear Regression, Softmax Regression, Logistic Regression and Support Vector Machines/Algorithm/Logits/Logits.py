from Read_Split_Data import SplitDataSet
import numpy as np

def Logits(lambda_=0.1):  # 正则化参数默认为0.1
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def log_loss(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    class LogisticRegression:
        def __init__(self, lr=0.01, num_iters=1000, lambda_=0.1):  # 添加正则化参数lambda_
            self.lr = lr
            self.num_iters = num_iters
            self.lambda_ = lambda_  # 正则化参数
            self.weights = None
            self.bias = None

        def fit(self, X, y):
            m, n = X.shape
            self.weights = np.zeros(n)
            self.bias = 0

            for _ in range(self.num_iters):
                y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
                dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (self.lambda_ / m) * self.weights  # 加入L2正则化项
                db = (1 / m) * np.sum(y_pred - y)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

        def predict(self, X):
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
            return (y_pred > 0.5).astype(int)

        def score(self, X, y):
            y_pred = self.predict(X)
            return np.mean(y_pred == y)

    model = LogisticRegression(lr=0.01, num_iters=1000, lambda_=lambda_)  # 使用正则化参数lambda_
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    def calculate_accuracy(y_true, y_pred):
        # 将预测值转换为二分类结果
        y_pred_binary = (y_pred > 0.5).astype(int)  # 将预测值大于 0.5 的置为 1，否则置为 0
        # 计算准确率
        accuracy = np.mean(y_true == y_pred_binary)  # 计算准确率
        # 计算预测正确和错误的数量
        correct_predictions = np.sum(y_true == y_pred_binary)  # 计算预测正确的数量
        incorrect_predictions = np.sum(y_true != y_pred_binary)  # 计算预测错误的数量
        return accuracy, correct_predictions, incorrect_predictions

    accuracy, correct_predictions, incorrect_predictions = calculate_accuracy(y_test,y_pred)  # 计算预测的准确率、正确和错误的预测数量
    print("正确预测数据:", correct_predictions)
    print("错误预测数据:", incorrect_predictions)
    print("准确率:", accuracy)

    return accuracy

# 调用 Logits() 函数来执行训练和评估，并返回模型准确率
accuracy = Logits()

