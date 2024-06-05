import numpy as np
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import scale

def readBooks():
    file = 'D:/Desktop/大二·下/统计学习方法/课内实践项目---1/datasets/dvd.svmlight'
    X, y = load_svmlight_files([file])
    X = X.toarray() # 将稀疏矩阵转换为密集矩阵
    y[y == -1] = 0  # 将标签中的-1替换为0，将类别标签转换为0和1
    X = scale(X) # 对特征矩阵 X 进行标准化,使得每个特征的均值为 0,方差为 1,进行特征缩放
    return X, y
