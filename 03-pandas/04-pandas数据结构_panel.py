import numpy as np
import pandas as pd

# Panel 和 MultiIndex是一样的
# Pandas从版本0.20.0开始弃用：推荐的用于表示3D数据的方法是通过DataFrame上的MultiIndex方法
if __name__ == '__main__':
    # Panel的创建
    # pandas.Panel(data=None, items=None, major_axis=None, minor_axis=None)
    # 作用：存储3维数组的Panel结构
    # 参数：
    # data: ndarray或者dataframe
    # items: 索引或类似数组的对象，axis = 0
    # major_axis: 索引或类似数组的对象，axis = 1
    # minor_axis: 索引或类似数组的对象，axis = 2
    p = pd.Panel(data=np.arange(24).reshape(4, 3, 2),
                 items=list('ABCD'),
                 major_axis=pd.date_range('20130101', periods=3),
                 minor_axis=['first', 'second'])
    print(p["A", :, :])
    print(p["D", :,  "first"])
