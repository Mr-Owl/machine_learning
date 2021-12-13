import numpy as np
import pandas as pd

if __name__ == '__main__':
    # DataFrame
    # 是一个类似于二维数组或表格(如excel)的对象，既有行索引，又有列索引
    # 行索引，表明不同行，横向索引，叫index，0轴，axis=0
    # 列索引，表名不同列，纵向索引，叫columns，1轴，axis=1
    # 创建方法
    # pd.DataFrame(data=None, index=None, columns=None)
    # 参数：
    # index：行标签。如果没有传入索引参数，则默认会自动创建一个从0-N的整数索引。
    # columns：列标签。如果没有传入索引参数，则默认会自动创建一个从0-N的整数索引。
    # 通过已有数据创建
    subjects = ["语文", "数学", "英语", "政治", "体育"]
    score = np.random.randint(40, 100, (10, 5))
    stu = ["同学" + str(i + 1) for i in range(score.shape[0])]
    score_df = pd.DataFrame(score, index=stu, columns=subjects)
    print(score_df)
    print()
    # DataFrame 属性    score = np.random.randint(40, 100, (10, 5))
    print(score_df.shape)  # 获取行列数
    print(score_df.index)  # 获取行标签
    print(score_df.columns)  # 获取列标签
    print(score_df.values)  # 获取所有值
    print(score_df.T)  # 对数据进行转置，即行列互换
    print(score_df.head())  # 显示前n行，默认为5
    print(score_df.head(1))
    print(score_df.tail())  # 显示后n行，默认为5
    print(score_df.tail(2))
    # 修改行列索引值 必须全部修改
    # score_df.index[0] = "你好"  # 报错
    stu = ["学生_" + str(i + 1) for i in range(score_df.shape[0])]
    score_df.index = stu
    print(score_df)
    subjects = ["语", "数", "英", "政", "体"]
    score_df.columns = subjects
    print(score_df)
    # DataFrame重设索引
    # 对象.reset_index(drop=False)
    # 设置新的下标索引
    # drop: 默认为False，不删除原来索引，如果为True, 删除原来的索引值
    # 修改后会返回修改后的内容
    print(score_df.reset_index())
    print(score_df.reset_index(drop=True))
    # 以某列值设置为新的索引
    # 对象.set_index(keys, drop=True)
    # keys : 列索引名成或者列索引名称的列表
    # drop : boolean, default True.当做新的索引，删除原来的列

    df = pd.DataFrame({'month': [1, 4, 7, 10],
                       'year': [2012, 2014, 2013, 2014],
                       'sale': [55, 40, 84, 31]})
    print(df)
    print(df.set_index("year"))
    # 设置多个索引，这样DataFrame就变成了一个具有MultiIndex的DataFrame
    print(df.set_index(["year", "month"]))

