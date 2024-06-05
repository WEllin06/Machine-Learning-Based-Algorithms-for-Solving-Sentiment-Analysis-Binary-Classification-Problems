from Read_Split_Data import SplitDataSet
from sklearn.naive_bayes import GaussianNB
def Gaussian():
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()
    model = GaussianNB()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    # 在测试集上评估模型
    accuracy = model.score(X_test, y_test)
    return accuracy

# 调用 Gaussian() 函数来执行训练和评估，并返回模型准确率
accuracy = Gaussian()


