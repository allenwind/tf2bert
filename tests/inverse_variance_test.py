import numpy as np
import matplotlib.pyplot as plt
from tf2bert.layers import MaskedMinVariancePooling

# 测试最小方差加权平均的正确性

func = MaskedMinVariancePooling(return_scores=True)

# 模仿十个随机漫步
x = np.cumsum(np.random.normal(size=(1, 10, 512)), axis=-1)

# 最小方差组合
y, w = func(x)

plt.plot(x[0].T, color="blue")
plt.plot(y[0], color="red")
print(w)
plt.show()
