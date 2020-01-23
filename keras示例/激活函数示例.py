# coding:utf-8
"""
激活函数示例
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.activations import sigmoid
from keras.activations import tanh
from keras.backend import get_value
from keras.activations import relu

x = np.arange(-5, 5, 0.1)
y_sigmoid = get_value(sigmoid(x))
plt.title(label="Sigmoid")
plt.plot(x, y_sigmoid)
plt.show()

y_tanh = get_value(tanh(x))
plt.title(label="Tanh")
plt.plot(x, y_tanh)
plt.show()

y_relu = get_value(relu(x))
plt.title(label="relu")
plt.plot(x, y_relu)
plt.show()
