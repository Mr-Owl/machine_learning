import numpy as np
import matplotlib.pyplot as plt
import random
from pylab import mpl

# 设置显示中文字体
# 字体需要额外下载安装，另外配置后需要去家目录cache清除缓存
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
def main():
    # 0.准备数据
    x = np.linspace(-10, 10, 1000)
    y = np.sin(x)

    # 1.创建画布
    plt.figure(figsize=(20, 8), dpi=100)

    # 2.绘制函数图像
    plt.plot(x, y)
    # 2.1 添加网格显示
    plt.grid()

    # 3.显示图像
    plt.show()


if __name__ == '__main__':
    main()