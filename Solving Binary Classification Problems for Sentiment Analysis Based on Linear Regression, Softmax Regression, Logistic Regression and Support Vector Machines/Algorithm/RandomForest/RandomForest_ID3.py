import numpy as np
from collections import Counter
from Read_Split_Data import SplitDataSet

class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _entropy(self, y):
        counter = Counter(y)
        probabilities = [count / len(y) for count in counter.values()]
        entropy = -np.sum([p * np.log2(p) for p in probabilities])
        return entropy

    def _information_gain(self, X, y, feature_index):
        entropy_parent = self._entropy(y)
        unique_values = np.unique(X[:, feature_index])
        entropy_children = 0
        for value in unique_values:
            subset_indices = np.where(X[:, feature_index] == value)
            subset_y = y[subset_indices]
            entropy_children += (len(subset_y) / len(y)) * self._entropy(subset_y)
        information_gain = entropy_parent - entropy_children
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
        stack = [(X, y, depth, {})]  # 使用栈来实现迭代构建决策树
        tree = {}  # 初始化树结构
        while stack:
            X, y, depth, node = stack.pop()  # 从栈中取出当前节点的信息
            if len(np.unique(y)) == 1:
                node['class'] = int(y[0])  # 如果所有样本属于同一类别，则返回叶子节点的类别
            elif self.max_depth is not None and depth >= self.max_depth:
                node['class'] = int(np.argmax(np.bincount(y.astype(int))))  # 如果树的深度达到最大深度限制，则返回叶子节点的类别
            else:
                best_feature_index = self._best_split(X, y)
                if best_feature_index is None:
                    node['class'] = int(np.argmax(np.bincount(y.astype(int))))  # 如果无法继续分裂，则返回叶子节点的类别
                else:
                    node['feature_index'] = best_feature_index
                    node['children'] = {}
                    unique_values = np.unique(X[:, best_feature_index])
                    for value in unique_values:
                        subset_indices = np.where(X[:, best_feature_index] == value)
                        subset_X = X[subset_indices]
                        subset_y = y[subset_indices]
                        node['children'][value] = {}
                        stack.append((subset_X, subset_y, depth + 1, node['children'][value]))  # 将子节点信息压入栈中
            if not tree:
                tree = node  # 将树的根节点设置为当前节点
        return tree

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i, sample in enumerate(X):
            predictions[i] = self._predict_sample(sample, self.tree)
        return predictions

    def _predict_sample(self, sample, tree):
        if 'class' in tree:
            return tree['class']
        feature_index = tree['feature_index']
        value = sample[feature_index]
        if value in tree['children']:
            return self._predict_sample(sample, tree['children'][value])
        else:
            children_trees = [tree['children'][child] for child in tree['children'] if isinstance(child, float)]
            if children_trees:
                return self._predict_sample(sample, children_trees[0])
            else:
                return np.argmax(np.bincount(list(tree['children'].values())[0].keys()))

class RandomForestID3:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            model = DecisionTreeID3(max_depth=self.max_depth)
            sample_indices = np.random.choice(len(X), len(X), replace=True)  # 使用有放回抽样来选择样本
            feature_indices = np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False)
            model.fit(X[sample_indices][:, feature_indices], y[sample_indices])
            self.models.append((model, feature_indices))

    def predict(self, X):
        if not self.models:
            return np.array([])
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, (model, feature_indices) in enumerate(self.models):
            predictions[:, i] = model.predict(X[:, feature_indices])
        return np.array([majority_vote(row) for row in predictions])

def majority_vote(predictions):
    if not predictions.size:
        return None
    counts = Counter(predictions)
    return counts.most_common(1)[0][0]

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)  # 计算正确分类的样本数
    total = len(y_true)  # 总样本数
    accuracy = correct / total  # 准确率
    return accuracy, correct, total - correct  # 返回准确率、正确分类样本数和错误分类样本数

def train_and_evaluate(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    model = RandomForestID3(n_estimators=n_estimators, max_depth=max_depth)  # 初始化随机森林模型
    model.fit(X_train, y_train)  # 在训练集上拟合模型
    y_pred = model.predict(X_test)  # 在测试集上进行预测
    accuracy, correct, incorrect = calculate_accuracy(y_test, y_pred)  # 计算准确率和正确、错误分类的样本数
    return accuracy, correct, incorrect  # 返回准确率、正确分类样本数和错误分类样本数

def main():
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()  # 分割数据集
    accuracy, correct, incorrect = train_and_evaluate(X_train, X_test, y_train, y_test)  # 训练并评估模型
    print("Accuracy:", accuracy)  # 打印准确率
    print("Correctly classified samples:", correct)  # 打印正确分类的样本数
    print("Incorrectly classified samples:", incorrect)  # 打印错误分类的样本数
    print("Ratio of correctly classified samples:", correct / (correct + incorrect))  # 打印错误分类的样本数与总样本数的比值

if __name__ == "__main__":
    main()  # 主函数入口
