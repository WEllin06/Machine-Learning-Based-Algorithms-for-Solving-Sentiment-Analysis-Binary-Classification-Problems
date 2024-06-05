### SplitDataSet.py
from Read_Split_Data import ReadData
from sklearn.model_selection import train_test_split

def SplitDataSetForSort():
    X, y = ReadData.readBooks()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=66)

    return X_train, X_test, y_train, y_test


