import numpy as np  # 数组和矩阵运算
from collections import Counter  # 计数
from sklearn.metrics import accuracy_score  # 准确率评估
from sklearn.model_selection import train_test_split  # 数据集划分
from Read_Split_Data import ReadData  # 读取数据函数

def SplitDataSetForSort():
    X, y = ReadData.readBooks()  # 读取数据
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=99)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.50, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

class DecisionTreeGini:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth  # 最大深度
        self.min_samples_split = min_samples_split  # 最小样本数

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)  # 构建决策树

    def _gini(self, y):
        # 计算基尼指数
        counter = Counter(y)
        probabilities = [count / len(y) for count in counter.values()]
        gini = 1 - np.sum([p**2 for p in probabilities])
        return gini

    def _gini_index(self, X, y, feature_index):
        # 计算基尼指数
        gini_parent = self._gini(y)
        unique_values = np.unique(X[:, feature_index])
        gini_children = 0
        for value in unique_values:
            subset_indices = np.where(X[:, feature_index] == value)
            subset_y = y[subset_indices]
            gini_children += (len(subset_y) / len(y)) * self._gini(subset_y)
        gini_index = gini_parent - gini_children
        return gini_index

    def _best_split(self, X, y):
        # 找到最佳划分特征
        best_feature_index = None
        best_gini_index = np.inf
        n_features = X.shape[1]
        for feature_index in range(n_features):
            gini_index = self._gini_index(X, y, feature_index)
            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_feature_index = feature_index
        return best_feature_index

    def _build_tree(self, X, y, depth=0):
        # 递归构建决策树
        if len(X) < self.min_samples_split:
            # 终止条件1: 最小样本数
            return {'class': int(np.argmax(np.bincount(y.astype(int))))}
        if self.max_depth is not None and depth >= self.max_depth:
            # 终止条件2: 最大深度
            return {'class': int(np.argmax(np.bincount(y.astype(int))))}
        best_feature_index = self._best_split(X, y)
        if best_feature_index is None:
            # 终止条件3: 无法再分割
            return {'class': int(np.argmax(np.bincount(y.astype(int))))}
        tree = {'feature_index': best_feature_index}
        unique_values = np.unique(X[:, best_feature_index])
        for value in unique_values:
            subset_indices = np.where(X[:, best_feature_index] == value)
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]
            tree[value] = self._build_tree(subset_X, subset_y, depth=depth + 1)
        return tree

    def predict(self, X):
        # 预测函数
        predictions = np.zeros(X.shape[0])
        for i, sample in enumerate(X):
            predictions[i] = self._predict_sample(sample, self.tree)
        return predictions

    def _predict_sample(self, sample, tree):
        # 递归预测
        if 'class' in tree:
            return tree['class']
        feature_index = tree['feature_index']
        value = sample[feature_index]
        if value in tree:
            return self._predict_sample(sample, tree[value])
        else:
            children_trees = [tree[child] for child in tree if isinstance(child, float)]
            if children_trees:
                return self._predict_sample(sample, children_trees[0])
            else:
                return np.argmax(np.bincount(list(tree.values())[1].keys()))

class RandomForestGini:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators  # 树的数量
        self.max_depth = max_depth  # 树的最大深度
        self.models = []

    def fit(self, X, y):
        # 训练模型
        for _ in range(self.n_estimators):
            model = DecisionTreeGini(max_depth=self.max_depth)
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            feature_indices = np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False)
            model.fit(X[sample_indices][:, feature_indices], y[sample_indices])
            self.models.append((model, feature_indices))

    def predict(self, X):
        # 预测函数
        if not self.models:
            return np.array([])
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, (model, feature_indices) in enumerate(self.models):
            predictions[:, i] = model.predict(X[:, feature_indices])
        return np.array([majority_vote(row) for row in predictions])

def majority_vote(predictions):
    # 投票函数
    if not predictions.size:
        return None
    counts = Counter(predictions)
    return counts.most_common(1)[0][0]

def main():
    # 数据集划分
    X_train, X_val, X_test, y_train, y_val, y_test = SplitDataSetForSort()
    # 模型训练和评估
    accuracy, correct, incorrect = train_and_evaluate(X_train, X_test, y_train, y_test)
    # 打印结果
    print("Accuracy:", accuracy)
    print("Correctly classified samples:", correct)
    print("Incorrectly classified samples:", incorrect)
    print("Ratio of correctly classified samples:", correct / len(y_test))

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # 模型训练和评估
    model = RandomForestGini(n_estimators=100, max_depth=10)  # 创建模型
    model.fit(X_train, y_train)  # 在训练集上拟合模型
    predictions = model.predict(X_test)  # 在测试集上进行预测
    accuracy = accuracy_score(y_test, predictions)  # 计算准确率
    correct = np.sum(predictions == y_test)  # 正确分类的样本数
    incorrect = len(y_test) - correct  # 错误分类的样本数
    return accuracy, correct, incorrect

if __name__ == "__main__":
    main()  # 主函数入口
