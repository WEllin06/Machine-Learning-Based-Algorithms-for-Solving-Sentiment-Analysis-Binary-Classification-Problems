import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Read_Split_Data import ReadData

# 假设X是数据矩阵，y是标签向量，lambda_是正则化参数
def compute_hinge_loss_gradient(w, b, X, y, lambda_):
    n_samples, n_features = X.shape
    distances = 1 - y * (np.dot(X, w) + b)
    dw = np.zeros(n_features)
    db = 0

    for i, d in enumerate(distances):
        if max(0, d) == 0:
            di = 0
        else:
            di = -y[i] * X[i]
            db -= y[i]
        dw += di
    dw = dw / n_samples + 2 * lambda_ * w  # 加上正则化项的梯度
    db = db / n_samples
    return dw, db


# 梯度下降算法
def gradient_descent(X, y, lr, epochs, lambda_):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    for _ in range(epochs):
        dw, db = compute_hinge_loss_gradient(w, b, X, y, lambda_)
        w -= lr * dw
        b -= lr * db
    return w, b

def predict(X, w, b):
    y_pred = np.sign(np.dot(X, w) + b)
    return y_pred

# 示例参数
lr = 0.01  # 学习率
epochs = 5000  # 代次数
lambda_ = 0.01  # 正则化参数

X,y=ReadData.readBooks()
y = np.where(y == 0, -1, y)
random_seed = 42

print(y)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加偏置项
X_train_scaled = np.hstack((np.ones((len(X_train_scaled), 1)), X_train_scaled))
X_test_scaled = np.hstack((np.ones((len(X_test_scaled), 1)), X_test_scaled))

# 假设X_train和y_train是你的训练数据和标签
w, b = gradient_descent(X_train_scaled, y_train, lr, epochs, lambda_)
predictions = predict(X_test_scaled, w, b)
print(predictions)

# 计算准确率
accurancy = np.mean(predictions == y_test)
print("最优准确率：", accurancy)
