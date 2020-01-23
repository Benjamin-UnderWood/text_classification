# coding:utf-8
"""
初始化器示例
"""

from keras.initializers import random_uniform
import numpy as np
import matplotlib.pyplot as plt
from keras.backend import get_value

x = range(300)
r = random_uniform(minval=0, maxval=1/np.sqrt(len(x)), seed=None)
y = get_value(r((300, )))


plt.title(label="uniform")
plt.plot(x, y)
plt.show()
