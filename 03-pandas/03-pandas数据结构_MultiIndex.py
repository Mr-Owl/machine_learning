import numpy as np
import pandas as pd

# MultiIndex
# MultiIndex是三维的数据结构;
# 多级索引（也称层次化索引）是pandas的重要功能
# 可以在Series、DataFrame对象上拥有2个以及2个以上的索引。
if __name__ == '__main__':
    df = pd.DataFrame({'month': [1, 4, 7, 10],
                       'year': [2012, 2014, 2013, 2014],
                       'sale': [55, 40, 84, 31]})
    print(df)
    # 设置多个索引此时DataFrame就变成了MultiIndex
    df = df.set_index(["year", "month"])
    # MultiIndex索引:levels存储的是多个索引值,labels是索引值的下标,names是levels的名称
    print(df.index)
    print(df.index.names)
    print(df.index.levels)
    print("--------------------")
    # 通过二维数组创建MultiIndex
    # pd.MultiIndex.from_arrays(二维数组, names=(第一列数据的名称, 第二列数据的名称))
    arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
    multi = pd.MultiIndex.from_arrays(arrays, names=("数量", "颜色"))
    print(multi)
