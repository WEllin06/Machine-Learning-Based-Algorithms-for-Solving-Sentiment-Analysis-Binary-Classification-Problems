from sklearn.datasets import load_svmlight_file

# 加载 svmlight 文件并获取数据维度
file_paths = [
    'D:/Desktop/大二·下/统计学习方法/课内实践项目---1/datasets/books.svmlight',
    'D:/Desktop/大二·下/统计学习方法/课内实践项目---1/datasets/dvd.svmlight',
    'D:/Desktop/大二·下/统计学习方法/课内实践项目---1/datasets/electronics.svmlight'
]

for file_path in file_paths:
    X, _ = load_svmlight_file(file_path)
    num_samples, num_features = X.shape
    print("文件 '{}' 的维度：{} 行 x {} 列".format(file_path, num_samples, num_features))
