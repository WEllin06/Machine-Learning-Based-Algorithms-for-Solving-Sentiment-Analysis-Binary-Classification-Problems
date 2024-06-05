import numpy as np
import scipy.optimize as opt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Store data into the variables
DATA = load_breast_cancer()
X, y = DATA.data, DATA.target
y[y==0] = -1
print(X.shape, y.shape)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print('training set:', X_train.shape, y_train.shape)
print('test set:', X_test.shape, y_test.shape)

# Preprocess the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

X_train = np.hstack((np.ones((len(X_train),1)), X_train))
X_test = np.hstack((np.ones((len(X_test),1)), X_test))
m_te, dim = X_test.shape

gamma = 0.01
K_train, K_test = rbf_kernel(X_train, X_test), rbf_kernel(X_test, X_test)

# define the objective function
def obj(alpha):
    cost = np.mean((K_train.dot(alpha))**2) - 2 * np.mean(K_test.dot(alpha)) + gamma * np.sum(alpha**2)
    return cost

# the initial point for optimization should be small to avoid overflow
init_params = 1e-3 * np.random.randn(m_te)
# learn the parameters
res = opt.minimize(obj, init_params, method='L-BFGS-B') # or use the SLSQP algorithm
alpha_hat = np.maximum(res.x, 0)

# hinge loss

lamda = 0.01

# define the objective function
def obj(theta):
    cost = np.mean(K_train.dot(alpha_hat) *  np.maximum(0, 1 - y_train * X_train.dot(theta))) + lamda * np.sum(theta**2)
    return cost

# the initial point for optimization should be small to avoid overflow
init_params = 1e-3 * np.random.randn(dim)
# learn the parameters
res = opt.minimize(obj, init_params, method='L-BFGS-B') # or use the SLSQP algorithm
theta_hat = res.x

# evaluate the learned model on testing data
FX_test = X_test.dot(theta_hat)
y_hat = np.where(FX_test<=0.,-1,1)
correct = (y_hat == y_test).sum()
print ('number of correct predictions: ',correct, ', prediction accuracy: ', correct / len(y_test))
# logistic loss

lamda = 0.01

# define the objective function
def obj(theta):
    cost = np.mean(K_train.dot(alpha_hat) *  np.log(1 + np.exp(-y_train * X_train.dot(theta)))) + lamda * np.sum(theta**2)
    return cost

# the initial point for optimization should be small to avoid overflow
init_params = 1e-3 * np.random.randn(dim)
# learn the parameters
res = opt.minimize(obj, init_params, method='L-BFGS-B') # or use the SLSQP algorithm
theta_hat = res.x

# evaluate the learned model on testing data
FX_test = X_test.dot(theta_hat)
y_hat = np.where(FX_test<=0., -1, 1)
correct = (y_hat == y_test).sum()
print ('number of correct predictions: ', correct, ', prediction accuracy: ', correct / len(y_test))