import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成一系列 x 值
x_values = np.linspace(-10, 10, 100)

# 计算相应的 Sigmoid 值
sigmoid_values = sigmoid(x_values)

# 绘制图形
plt.plot(x_values, sigmoid_values, label='Sigmoid Function')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.grid(True)
plt.legend()

# 添加坐标轴上的交点
intersection_x = 0
intersection_y = sigmoid(intersection_x)
plt.scatter(intersection_x, intersection_y, color='red', label=f'Intersection at ({intersection_x}, {intersection_y:.1f})')

plt.legend()
plt.show()
