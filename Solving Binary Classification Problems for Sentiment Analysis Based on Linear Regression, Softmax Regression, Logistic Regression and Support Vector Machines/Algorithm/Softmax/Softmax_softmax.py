import numpy as np
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# 定义 Softmax 函数
def softmax(z):
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# 定义损失函数
def loss_function(W, X, Y):
    N = X.shape[0]  # 获取样本数量
    M = np.dot(X, W)
    probabilities = softmax(M)
    loss = -np.sum(Y * np.log(probabilities + 1e-12)) / N  # 加入一个微小值以避免 log(0) 的情况，并归一化
    return loss

# 定义梯度函数
def gradient(W, X, Y):
    N = X.shape[0]  # 获取样本数量
    M = np.dot(X, W)
    probabilities = softmax(M)
    gradient = np.dot(X.T, probabilities - Y) / N  # 归一化
    return gradient

# 使用梯度下降算法优化参数
def train_softmax_regression(X_train, Y_train, learning_rate=0.01, num_epochs=1000):
    dim = X_train.shape[1]  # 特征维度
    n_class = 2  # 二分类问题
    W = np.zeros((dim, n_class))  # 初始化权重矩阵为零向量
    for epoch in range(num_epochs):
        grad = gradient(W, X_train, Y_train)
        W -= learning_rate * grad
        if epoch % 100 == 0:
            loss = loss_function(W, X_train, Y_train)
            print("Epoch {}: Loss {}".format(epoch, loss))
    return W

# 计算准确率以及成功分类的样本总数和未成功被分类的样本总数
def calculate_accuracy_and_counts(X, Y, W):
    M = np.dot(X, W)
    probabilities = softmax(M)
    predictions = np.argmax(probabilities, axis=1)  # 预测的类别
    actual = np.argmax(Y, axis=1)  # 实际的类别
    accuracy = np.mean(predictions == actual)  # 计算准确率
    correct_count = np.sum(predictions == actual)  # 计算成功分类的样本总数
    error_count = len(predictions) - correct_count  # 计算未成功被分类的样本总数
    return accuracy, correct_count, error_count

def readBooks():
    file = 'D:/Desktop/大二·下/统计学习方法/课内实践项目---1/datasets/electronics.svmlight'
    X, y = load_svmlight_files([file])
    X = X.toarray()  # 将稀疏矩阵转换为密集矩阵
    X = scale(X, with_mean=False) # 只进行特征缩放，不进行均值中心化
    return X, y

# 准备数据
X, y = readBooks()

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66)

# 将标签转换为整数类型，并从0开始编号
classes = np.unique(y_train)
class_mapping = {cls: i for i, cls in enumerate(classes)}
y_train_mapped = np.array([class_mapping[cls] for cls in y_train])
y_test_mapped = np.array([class_mapping[cls] for cls in y_test])

# 将标签进行独热编码
n_class = len(classes)
Y_train = np.eye(n_class)[y_train_mapped]
Y_test = np.eye(n_class)[y_test_mapped]

# 训练 Softmax 回归模型
W = train_softmax_regression(X_train, Y_train)

# 在测试集上评估准确率，并计算成功分类的样本总数和未成功被分类的样本总数
accuracy, correct_count, error_count = calculate_accuracy_and_counts(X_test, Y_test, W)
print("Correctly classified samples:", correct_count)
print("Misclassified samples:", error_count)
print("Accuracy on test set:", accuracy)

