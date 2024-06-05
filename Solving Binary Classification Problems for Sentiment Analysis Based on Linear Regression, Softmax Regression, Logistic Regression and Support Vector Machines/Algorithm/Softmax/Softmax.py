import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# 生成一组输入数据
x = np.linspace(-10, 10, 100)
# 计算softmax输出
y = softmax(x)

# 绘制函数图像
plt.plot(x, y)
plt.title('Softmax Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()
