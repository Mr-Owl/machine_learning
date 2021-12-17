import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv("./data/stock_day.csv")
    # print(data.index)  # 这里面的日期object对象
    # 将字符串日期改成日期类型 这样子就可以对时间进行取星期几,几号等操作了
    time = pd.to_datetime(data.index)
    # print(time.weekday)
    data["week"] = time.weekday  # 新增一列数据
    # 增加p_n列代表涨跌,值:修改data中p_change大于0为1为涨,小于等于0为0为跌
    data["p_n"] = np.where(data["p_change"] > 0, 1, 0)
    # print(data.head())
    # 交叉表：交叉表用于计算一列数据对于另外一列数据的分组个数(用于统计分组频率的特殊透视表)
    # 简单来说就是一列数据到另一列数据中的数量
    # pd.crosstab(value1, value2)
    count = pd.crosstab(data["week"], data["p_n"])
    # print(count)
    # 对于每个星期一等的总天数求和，运用除法运算求出比例
    count_sum = count.sum(axis=1)
    # print(count_sum)
    ret = count.div(count_sum, axis=0)  # 计算百分比
    # print(ret)
    # ret.plot(kind="bar", stacked=True)
    # plt.show()

    # 透视表：透视表是将原有的DataFrame的列分别作为行索引和列索引，然后对指定的列应用聚集函数
    # 简单的来说就是一列数据对于另外一列数据的百分占比
    # data.pivot_table(）
    # DataFrame.pivot_table([], index=[])
    print(data.pivot_table(["p_n"], index="week"))