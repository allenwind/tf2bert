import numpy as np
import matplotlib.pyplot as plt
from tf2bert.layers import MaskedMinVariancePooling

# 测试最小方差加权平均的正确性

func = MaskedMinVariancePooling(return_scores=True)

# 模仿二十个随机漫步
x = np.cumsum(np.random.normal(0.1, 1, size=(1, 20, 1024)), axis=-1)

# 直接平均
ym = np.mean(x, axis=1)
ym_var = np.var(ym)

# 最小方差组合
y, w = func(x)
y_var = np.var(y)

plt.plot(x[0].T, color="blue")
plt.plot(y[0], color="red", label=f"inverse variance sum, var={y_var:.2f}")
plt.plot(ym[0], color="green", label=f"mean sum, var={ym_var:.2f}")
plt.legend(loc="upper left")
print(w)
plt.show()
