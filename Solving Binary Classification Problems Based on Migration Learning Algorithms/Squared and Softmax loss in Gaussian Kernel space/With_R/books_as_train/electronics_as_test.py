import numpy as np
import scipy.optimize as opt
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_files

file = '/datasets/books.svmlight'
# 读取SVMLight格式的数据文件
X, y = load_svmlight_files([file])
# 将稀疏矩阵转换为密集矩阵,将稀疏矩阵 X 转换为标准的 NumPy 数组
X = X.toarray()
# 加载数据
X_Train, Y_Train = X,y
Y_Train[Y_Train==0] = -1


file = '/datasets/electronics.svmlight'
# 读取SVMLight格式的数据文件
X, y = load_svmlight_files([file])
# 将稀疏矩阵转换为密集矩阵,将稀疏矩阵 X 转换为标准的 NumPy 数组
X = X.toarray()
X_Test, Y_Test = X,y
Y_Test[Y_Test==0] = -1


# 数据预处理
scaler = StandardScaler()
scaler.fit(X_Train)
X_train ,X_test = scaler.transform(X_Train),scaler.transform(X_Test)

# 添加偏置项
X_train = np.hstack((np.ones((len(X_train), 1)), X_train))
X_test = np.hstack((np.ones((len(X_test), 1)), X_test))
m_te, dim = X_test.shape

# 计算核矩阵
gamma = 0.01  # 调整 gamma 值
K_train, K_test = rbf_kernel(X_train, X_test), rbf_kernel(X_test, X_test)

# 求解参数 alpha
def obj(alpha):
    cost = np.mean((K_train.dot(alpha))**2) - 2 * np.mean(K_test.dot(alpha)) + gamma * np.sum(alpha**2)  # 调整正则化参数
    return cost

init_alpha = 1e-3 * np.random.randn(m_te)
result_alpha = opt.minimize(obj, init_alpha, method='L-BFGS-B')
alpha_hat = np.maximum(result_alpha.x, 0)


# 1. Square Loss
lamda = 0.01  # 调整正则化参数
def obj(theta):
    cost = np.mean(K_train.dot(alpha_hat) * (1 - Y_Train * X_train.dot(theta))**2) + lamda * np.sum(theta**2)
    return cost

init_theta = 1e-3 * np.random.randn(dim)
result_theta = opt.minimize(obj, init_theta, method='L-BFGS-B')
theta_hat = result_theta.x

FX_test = X_test.dot(theta_hat)
y_hat = np.where(FX_test <= 0., -1, 1)
FX_test = X_test.dot(theta_hat)
y_hat = np.where(FX_test<=0.,-1,1)
correct = (y_hat == Y_Test).sum()
print('预测准确率 (Square Loss):', correct / len(Y_Test))


# 2. Logistic Loss
lamda = 0.01  # 调整正则化参数
def obj(theta):
    cost = np.mean(K_train.dot(alpha_hat) *  np.log(1 + np.exp(-Y_Train * X_train.dot(theta)))) + lamda * np.sum(theta**2)
    return cost

init_theta =  1e-3 * np.random.randn(dim)
result_theta = opt.minimize(obj, init_theta, method='L-BFGS-B')
theta_hat = result_theta.x

FX_test = X_test.dot(theta_hat)
y_hat = np.where(FX_test <= 0., -1, 1)
y_hat = np.where(FX_test<=0.,-1,1)
correct = (y_hat == Y_Test).sum()
print('预测准确率 (Logistic Loss):', correct / len(Y_Test))


# 3. Hinge Loss
lamda = 0.01  # 调整正则化参数
def obj(theta):
    cost = np.mean(K_train.dot(alpha_hat) * np.maximum(0, 1 - Y_Train * X_train.dot(theta))) + lamda * np.sum(theta**2)
    return cost

init_theta = 1e-3*np.random.randn(dim)
result_theta = opt.minimize(obj, init_theta, method='L-BFGS-B')
theta_hat = result_theta.x

FX_test = X_test.dot(theta_hat)
y_hat = np.where(FX_test<=0.,-1,1)
correct = (y_hat == Y_Test).sum()
print('预测准确率 (Hinge Loss):', correct / len(Y_Test))
