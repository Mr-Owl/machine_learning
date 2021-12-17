import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("./data/stock_day.csv")
    p_change = data["p_change"]
    bins = [-100, -7, -5, -3, 0, 3, 5, 7, 100]
    p_count = pd.cut(p_change, bins)  # 分组
    dummies = pd.get_dummies(p_count, prefix='rise')  # 哑变量矩阵 one-hot
    # 合并
    # 1 pd.concat实现数据合并
    # pd.concat([data1, data2], axis=1)
    # 按照行或列进行合并,axis=0为列索引，axis=1为行索引
    print(pd.concat([data, dummies], axis=1))
    print("---------------------------")
    # 2 pd.merge(left, right, how='inner', on=None)
    # 可以指定按照两组数据的共同键值对合并或者左右各自
    # left: DataFrame
    # right: 另一个DataFrame
    # on: 指定的共同键
    # how:按照什么方式连接
    left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                         'key2': ['K0', 'K1', 'K0', 'K1'],
                         'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})

    right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                          'key2': ['K0', 'K0', 'K0', 'K0'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})
    # how:inner 内连接 匹配左右都有的,任意一遍有相同多个的,匹配多次  默认值
    print(pd.merge(left, right, on=["key1", "key2"]))
    print("----------------------")
    # how:outer 外连接 左右都匹配. 任意一遍没有的用Nan填充
    print(pd.merge(left, right, on=["key1", "key2"], how="outer"))
    print("----------------------")
    # how:left 左连接 已左边为主,右边没有Nan填充
    print(pd.merge(left, right, on=["key1", "key2"], how="left"))
    print("----------------------")
    # how:right 右连接  已右边为主,左边没有Nan填充
    print(pd.merge(left, right, on=["key1", "key2"], how="right"))
    print("----------------------")
