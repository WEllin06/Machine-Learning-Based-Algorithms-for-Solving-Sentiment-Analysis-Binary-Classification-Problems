import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Read_Split_Data import ReadData


def SplitDataSetForSort():
    X, y = ReadData.readBooks()
    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=99)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.50, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

class DecisionTreeGini:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _gini(self, y):
        counter = Counter(y)
        probabilities = [count / len(y) for count in counter.values()]
        gini = 1 - np.sum([p**2 for p in probabilities])
        return gini

    def _gini_index(self, X, y, feature_index):
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


class RandomForestGini:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            model = DecisionTreeGini(max_depth=self.max_depth)
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


def random_search(X_train, X_val, X_test, y_train, y_val, y_test, n_iterations=100):
    best_accuracy = 0
    best_params = None

    for _ in range(n_iterations):
        n_estimators = np.random.randint(1, 20)
        max_depth = np.random.randint(1, 15)

        model = RandomForestGini(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}

    # Use the best model found in validation for evaluation on the test set
    best_model = RandomForestGini(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
    best_model.fit(X_train, y_train)
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return best_params, test_accuracy


def random_forest_example():
    X_train, X_val, X_test, y_train, y_val, y_test = SplitDataSetForSort()

    # Splitting the validation set for random search
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    # Random search for best parameters using validation set
    best_params, test_accuracy = random_search(X_train, X_train_val, X_test, y_train, y_train_val, y_test,
                                               n_iterations=100)

    print("Best parameters:", best_params)
    print("Accuracy on test set using best parameters:", test_accuracy)


random_forest_example()