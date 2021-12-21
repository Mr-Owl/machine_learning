import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 当处理一组数据时，通常先要做的就是了解变量是如何分布的。
# 对于单变量的数据来说 采用直方图或核密度曲线是个不错的选择，
# 对于双变量来说，可采用多面板图形展现，比如 散点图、二维直方图、核密度估计图形等。
# 针对这种情况， Seaborn库提供了对单变量和双变 量分布的绘制函数，如 displot()函数、 jointplot()函数，下面来介绍这些函数的使用，具体内容如下：
if __name__ == '__main__':
    np.random.seed(0)  # 确定随机数生成器生成的数据是一样的,每次生成的随机数一样
    arr = np.random.randn(100)
    # 1 绘制单变量分布
    # 可以采用最简单的直方图描述单变量的分布情况。
    # Seaborn中提供了 distplot()函数，它默认绘制的是一个带有核密度估计曲线的直方图。 distplot()函数的语法格式如下。
    # seaborn.distplot(a, bins=None, kde=True, rug=False, fit=None, color=None)
    # 上述函数中常用参数的含义如下：
    # (1) a：表示要观察的数据，可以是 Series、一维数组或列表。
    # (2) bins：用于控制条形的数量。
    # (3) kde：接收布尔类型，表示是否绘制高斯核密度估计曲线。
    # (4) rug：接收布尔类型，表示是否在支持的轴方向上绘制rugplot。
    sns.displot(arr, bins=10, kde=True, rug=True)
    plt.show()
