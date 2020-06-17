import numpy as np
import matplotlib.pyplot as plt

# 標準偏差
sigma = 1
"""
平均0, 分散1 の正規分布
"""
def normalDistribution(x):
    y = np.exp( -x**2 / (2*sigma**2) ) / np.sqrt( 2 * np.pi * sigma**2 )
    return y

x = np.arange(-4, 4, 0.0001)
# 横軸の変数。縦軸の変数。
plt.plot(x, normalDistribution(x))
# 描画実行
plt.show()