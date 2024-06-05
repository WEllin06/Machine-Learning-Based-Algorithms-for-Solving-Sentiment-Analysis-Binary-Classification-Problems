import numpy as np
from collections import Counter
from Read_Split_Data import SplitDataSet

class DecisionTreeC45:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _entropy(self, y):
        counter = Counter(y)
        probabilities = [count / len(y) for count in counter.values()]
        entropy = -np.sum([p * np.log2(p) for p in probabilities])
        return entropy

    def _information_gain_ratio(self, X, y, feature_index):
        entropy_parent = self._entropy(y)
        unique_values, counts = np.unique(X[:, feature_index], return_counts=True)
        entropy_children = 0
        split_info = 0

        for value, count in zip(unique_values, counts):
            subset_indices = np.where(X[:, feature_index] == value)
            subset_y = y[subset_indices]
            subset_entropy = self._entropy(subset_y)
            entropy_children += (count / len(y)) * subset_entropy
            split_info -= (count / len(y)) * np.log2(count / len(y))

        information_gain = entropy_parent - entropy_children
        information_gain_ratio = information_gain / split_info if split_info != 0 else 0
        return information_gain_ratio

    def _best_split(self, X, y):
        best_feature_index = None
        best_information_gain_ratio = -np.inf
        n_features = X.shape[1]

        for feature_index in range(n_features):
            information_gain_ratio = self._information_gain_ratio(X, y, feature_index)
            if information_gain_ratio > best_information_gain_ratio:
                best_information_gain_ratio = information_gain_ratio
                best_feature_index = feature_index

        return best_feature_index

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return {'class': int(y[0])}

        if self.max_depth is not None and depth >= self.max_depth:
            return {'class': int(np.argmax(np.bincount(y.astype(int))))}

        best_feature_index = self._best_split(X, y)
        if best_feature_index is None:
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
        predictions = np.zeros(X.shape[0])
        for i, sample in enumerate(X):
            predictions[i] = self._predict_sample(sample, self.tree)
        return predictions

    def _predict_sample(self, sample, tree):
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

class RandomForestC45:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            model = DecisionTreeC45(max_depth=self.max_depth)
            sample_indices = np.random.choice(len(X), len(X), replace=True)
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
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy, correct, total - correct

def train_and_evaluate(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    model = RandomForestC45(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, correct, incorrect = calculate_accuracy(y_test, y_pred)
    return accuracy, correct, incorrect

def main():
    X_train, X_test, y_train, y_test = SplitDataSet.SplitDataSetForSort()
    accuracy, correct, incorrect = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("Accuracy:", accuracy)
    print("Correctly classified samples:", correct)
    print("Incorrectly classified samples:", incorrect)
    print("Ratio of correctly classified samples:", correct / (correct + incorrect))

if __name__ == "__main__":
    main()
