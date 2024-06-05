import warnings
from Read_Split_Data import SplitDataSet
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore", category=FutureWarning)

def KNN():
    # 创建K最近邻分类器
    model = KNeighborsClassifier(n_neighbors=3)
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

    # 使用训练集拟合模型
    model.fit(X_train, y_train)
    model.score (X_test,y_test)
    accuracy = model.score(X_test, y_test)
    return accuracy

# 调用 KNN() 函数来执行训练和评估，并返回模型准确率
accuracy = KNN()