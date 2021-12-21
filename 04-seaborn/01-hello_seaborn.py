import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
    # 2 绘制双变量分布
    # 两个变量的二元分布可视化也很有用。在 Seaborn中最简单的方法是使用 jointplot()函数，该函数可以创建一个多面板图形，比如散点图、二维直方图、核密度估计等，以显示两个变量之间的双变量关系及每个变量在单坐标轴上的单变量分布。
    # jointplot()函数的语法格式如下。
    # seaborn.jointplot(x, y, data=None,
    #                   kind='scatter', stat_func=None, color=None,
    #                   ratio=5, space=0.2, dropna=True)
    # 上述函数中常用参数的含义如下：
    # (1) kind：表示绘制图形的类型。
    # (2) stat_func：用于计算有关关系的统计量并标注图(pearsonr在0-1之间,越大表示关系越大)。  新版本通过annotate方法写入通过scipy中的status的pearsonr方法设置
    # (3) color：表示绘图元素的颜色。
    # (4) size：用于设置图的大小(正方形)。
    # (5) ratio：表示中心图与侧边图的比例。该参数的值越大，则中心图的占比会越大。必须是整数
    # (6) space：用于设置中心图与侧边图的间隔大小。
    # 下面以散点图、二维直方图、核密度估计曲线为例，为大家介绍如何使用 Seaborn绘制这些图形。
    #
    # 3.1 绘制散点图
    # 创建DataFrame对象
    df = pd.DataFrame({"x": np.random.randn(500), "y": np.random.randn(500)})
    # 绘制散布图
    sns.jointplot("x", "y", data=df,
                  # 新版本stat_func=True已无法使用
                  kind="scatter", color="r", size=10,
                  ratio=8, space=1)
    plt.show()
    # 3.2 绘制二维直方图
    # 二维直方图类似于“六边形”图，主要是因为它显示了落在六角形区域内的观察值的计数，
    # 适用于较大的数据集。当调用 jointplot()函数时，只要传入kind="hex"，就可以绘制二维直方图，具体示例代码如下。
    sns.jointplot(x="x", y="y", data=df, kind="hex")
    plt.show()
    # 从六边形颜色的深浅，可以观察到数据密集的程度，另外，
    # 图形的上方和右侧仍然给出了直方图。注意，在绘制二维直方图时，最好使用白色背景。
    # 3.3 绘制核密度估计图形
    # 利用核密度估计同样可以查看二元分布，其用等高线图来表示。
    # 当调用jointplot()函数时只要传入ind="kde"，就可以绘制核密度估计图形，具体示例代码如下。
    sns.jointplot(x="x", y="y", data=df,
                  kind="kde",
                  shade=True)  # shade 开启深度颜色显示
    plt.show()
    # 通过观等高线的颜色深浅，可以看出哪个范围的数值分布的最多，
    # 哪个范围的数值分布的最少
    # 4 绘制成对的双变量分布
    # 要想在数据集中绘制多个成对的双变量分布，则可以使用pairplot()
    # 函数实现，该函数会创建一个坐标轴矩阵，并且显示Datafram对象中每对变量的关系。另外，pairplot()
    # 函数也可以绘制每个变量在对角轴上的单变量分布。
    # 接下来，通过sns.pairplot()
    # 函数绘制数据集变量间关系的图形，示例代码如下
    # 本次使用seaborn自带的iris鸢尾花数据集
    # 该数据集为seaborn内置的数据集,他会先在主目录seaborn-data去找,如果没有,就会去进行下载,可以
    # 先去https://github.com/mwaskom/seaborn-data下载之后放到seaborn-data中,防止该行请求报错
    dataset = sns.load_dataset("iris")
    print(dataset.head())
    sns.pairplot(dataset)
