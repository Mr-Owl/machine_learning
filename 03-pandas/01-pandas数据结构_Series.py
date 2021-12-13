import numpy as np
import pandas as pd

# 专门用于数据挖掘的开源python库
# 以Numpy为基础，借力Numpy模块在计算方面性能高的优势
# 基于matplotlib，能够简便的画图
if __name__ == '__main__':
    # 1 Series类似于一维数组的数据结构，它能够保存任何数据类型的数据
    # 主要由一组数据和与之相关的索引两部分构成
    # 创建：pd.Series(data=None, index=None, dtype=None)
    # data：传入的数据，可以是ndarray、list等
    # index：索引，必须是唯一的，且与数据的长度相等。如果没有传入索引参数，则默认会自动创建一个从0 - N的整数索引。
    # dtype：数据的类型
    print(pd.Series(np.arange(10)))
    print(pd.Series(np.random.randint(1, 10, 5), index=[1, 2, 3, 4, 5]))
    print(pd.Series({"apple": 1.5, "pea": 1, "banana": 2}))
    # Series属性
    # index 获取索引 对象.index
    fruit_price = pd.Series({"apple": 1.5, "pea": 1, "banana": 2})
    print(fruit_price.index)
    # values 获取所有的值 对象.values
    print(fruit_price.values)
    # 通过索引获取对应的值  对象[索引]
    print(fruit_price["apple"])
    print("--------------------------")

