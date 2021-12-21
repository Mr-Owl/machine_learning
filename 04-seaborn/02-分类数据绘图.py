import seaborn as sns
import matplotlib.pyplot as plt

"""
数据集中的数据类型有很多种，除了连续的特征变量之外，最常见的就是类别型的数据了，比如人的性别、学历、爱好等，这些数据类型都不能用连续的变量来表示，而是用分类的数据来表示。

Seaborn针对分类数据提供了专门的可视化函数，这些函数大致可以分为如下三种:

分类数据散点图: swarmplot()与 stripplot()。
类数据的分布图: boxplot() 与 violinplot()。
分类数据的统计估算图:barplot() 与 pointplot()。
"""

if __name__ == '__main__':
    # 加载内置数据集
    tips = sns.load_dataset('tips')
    # print(dataset.head())
    # 1 类别散点图
    # 通过 stripplot()函数可以画一个散点图， stripplot0函数的语法格式如下。
    # seaborn.stripplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, jitter=False)
    # 上述函数中常用参数的含义如下
    # (1) x，y，hue：用于绘制长格式数据的输入。 hue可以理解为在一个分类中在进行分类
    # 长数据一般指数据集中的变量没有做明确的细分,变量中至少有一个变量中的元素存在值严重重复循环的情况.
    # 一列包含了所有的变量,另一列则是与之相关的值
    # (2) data：用于绘制的数据集。如果x和y不存在，则它将作为宽格式，否则将作为长格式。
    # 宽数据 数据集对所有的变量进行了明确的细分,个变量的值不存在重复循环的情况,也无法归类
    # (3) jitter：表示抖动的程度(仅沿类別轴)。当很多数据点重叠时，可以指定抖动的数量或者设为Tue使用默认值。
    sns.stripplot(x="day", y="total_bill", data=tips, hue="time", jitter=True)
    plt.show()
    # 该函数所有的数据点都不会重叠，可以很清晰地观察到数据的分布情况，示例代码如下。
    sns.swarmplot(x="day", y="total_bill", data=tips, hue="time")
    plt.show()
    # 2 类别内的数据分布
    # 2.1 箱形图:
    # 箱形图（Box-plot）又称为盒须图、盒式图或箱线图，是一种用作显示一组数据分散情况资料的统计图。因形状如箱子而得名。
    # 箱形图于1977年由美国著名统计学家约翰·图基（John Tukey）发明。它能显示出一组数据的最大值、最小值、中位数、及上下四分位数。
    # seaborn中用于绘制箱形图的函数为 boxplot()，其语法格式如下:
    # seaborn.boxplot(x=None, y=None, hue=None,
    #                 data=None, orient=None, color=None,
    #                 saturation=0.75, width=0.8)
    # (1) palette：用于设置不同级别色相的颜色变量。---- palette=["r","g","b","y"]
    # (2) saturation：用于设置数据显示的颜色饱和度。---- 使用小数表示
    sns.boxplot("day", "total_bill", data=tips, hue="time", palette=["g", "r"], saturation=0.8)
    plt.show()
    # 2.1 小提琴图
    # 小提琴图 (Violin Plot) 用于显示数据分布及其概率密度。
    # 这种图表结合了箱形图和密度图的特征，主要用来显示数据的分布形状。
    # 中间的黑色粗条表示四分位数范围，从其延伸的幼细黑线代表 95% 置信区间，而白点则为中位数。
    # 箱形图在数据显示方面受到限制，简单的设计往往隐藏了有关数据分布的重要细节。例如使用箱形图时，我们不能了解数据分布。虽然小提琴图可以显示更多详情，但它们也可能包含较多干扰信息。
    # seaborn中用于绘制提琴图的函数为violinplot()，其语法格式如下
    # seaborn.violinplot(x=None, y=None, hue=None, data=None)
    sns.violinplot(x="day", y="total_bill", data=tips)
    plt.grid()
    plt.show()
    # 3 类别内的统计估计
    # 3.1
    # 绘制条形图
    # 最常用的查看集中趋势的图形就是条形图。默认情况下， barplot函数会在整个数据集上使用均值进行估计。
    # 若每个类别中有多个类别时(使用了hue参数)，则条形图可以使用引导来计算估计的置信区间
    # (是指由样本统计量所构造的总体参数的估计区间)，并使用误差条来表示置信区间。
    # 使用 barplot()
    # 函数的示例如下
    sns.barplot(x="day", y="total_bill", data=tips)  # 直方图上的细线代表误差条,平均值+-向上或向下的部分
    plt.show()
    # 3.2 绘制点图
    # 另外一种用于估计的图形是点图，可以调用 pointplot()函数进行绘制，
    # 该函数会用高度低计值对数据进行描述，而不是显示完整的条形，
    # 它只会绘制点估计和置信区间。
    # 通过 pointplot()函数绘制点图的示例如下。
    # sns.pointplot(x="day", y="total_bill", data=tips)
    sns.pointplot("day", "total_bill", data=tips)  # 并使用误差条来表示置信区间。和直方图一样,把直方图顶点作为结点
    plt.show()
