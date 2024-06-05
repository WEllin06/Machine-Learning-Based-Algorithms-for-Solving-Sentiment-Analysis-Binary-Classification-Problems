from Read_Split_Data import SplitDataSet
from sklearn.ensemble import RandomForestClassifier
def Random():
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    # 在测试集上评估模型
    accuracy = model.score(X_test, y_test)
    return accuracy
