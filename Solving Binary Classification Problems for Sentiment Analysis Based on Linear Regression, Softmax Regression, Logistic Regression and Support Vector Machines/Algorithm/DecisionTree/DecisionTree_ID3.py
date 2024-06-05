### 信息增益
import numpy as np
from Read_Split_Data import SplitDataSet


class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.tree = {}
        self.max_depth = max_depth

    def fit(self, X, y):
        y = y.astype(int)  # 将标签数据转换为整型
        self.tree = self._build_tree(X, y)

    def _entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            entropy -= p_cls * np.log2(p_cls)
        return entropy

    def _information_gain(self, X, y, feature_index):
        # Calculate parent entropy
        parent_entropy = self._entropy(y)

        # Calculate entropy after splitting by feature
        unique_values = np.unique(X[:, feature_index])
        split_entropy = 0
        for value in unique_values:
            subset_indices = np.where(X[:, feature_index] == value)
            subset_y = y[subset_indices]
            split_entropy += len(subset_y) / len(y) * self._entropy(subset_y)

        # Calculate information gain
        information_gain = parent_entropy - split_entropy
        return information_gain

    def _best_split(self, X, y):
        best_feature_index = None
        best_information_gain = -np.inf
        n_features = X.shape[1]

        for feature_index in range(n_features):
            information_gain = self._information_gain(X, y, feature_index)
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_index = feature_index

        return best_feature_index

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return {'class': y[0]}

        if X.shape[1] == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return {'class': np.argmax(np.bincount(y))}

        best_feature_index = self._best_split(X, y)
        if best_feature_index is None:
            return {'class': np.argmax(np.bincount(y))}

        tree = {'feature_index': best_feature_index}
        unique_values = np.unique(X[:, best_feature_index])
        for value in unique_values:
            subset_indices = np.where(X[:, best_feature_index] == value)
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]
            if len(subset_y) == 0:
                tree[value] = {'class': np.argmax(np.bincount(y))}
            else:
                tree[value] = self._build_tree(subset_X, subset_y, depth=depth+1)

        return tree

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._predict_sample(sample, self.tree))
        return np.array(predictions)

    def _predict_sample(self, sample, tree):
        if 'class' in tree:
            return tree['class']
        feature_index = tree['feature_index']
        value = sample[feature_index]
        if value in tree:
            return self._predict_sample(sample, tree[value])
        else:
            return np.argmax(np.bincount(list(tree.values())[1].keys()))

# 使用示例
def decision_tree_example():
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()

    # 创建并训练决策树模型
    model = DecisionTreeID3(max_depth=0)  # 设置最大深度
    model.fit(X_train, y_train)

    # 在测试集上评估模型
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    return accuracy

# 调用 decision_tree_example() 函数来执行训练和评估，并返回模型准确率
accuracy = decision_tree_example()
print("准确率：", accuracy)



