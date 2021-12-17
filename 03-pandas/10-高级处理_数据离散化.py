import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("./data/stock_day.csv")
    print(data.head())
    print("--------------------------")
    p_change = data["p_change"]
    print(p_change)
    # 连续属性离散化的目的是为了简化数据结构，数据离散化技术可以用来减少给定连续属性值的个数。
    # 离散化方法经常作为数据挖掘的工具。
    # 1 pd.qcut(data, q)：
    # 对数据进行分组将数据分组，一般会与value_counts搭配使用，统计每组的个数
    # series.value_counts()：统计分组次数
    qcut = pd.qcut(p_change, 10)
    print(qcut)
    print(qcut.value_counts())
    print("--------------------------")
    # 2 pd.cut(data, bins)
    # data 数据  bins分组
    bins = [-100, -7, -5, -3, 0, 3, 5, 7, 100]
    p_count = pd.cut(p_change, bins)
    print(p_count)
    print(p_count.value_counts())
    print("--------------------------")
    # 2 one-hot编码(哑变量矩阵)
    # 把每个类别生成一个布尔列，这些列中只有一列可以为这个样本取值为1.其又被称为独热编码
    # pandas.get_dummies(data, prefix=None)
    # data:array-like, Series, or DataFrame
    # prefix:分组名字
    dummies = pd.get_dummies(p_count, prefix='rise')
    print(dummies.head())
