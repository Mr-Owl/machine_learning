import pandas as pd


def main():
    # 读取csv数据
    data = pd.read_csv("./data/stock_day.csv")
    print(data.head())
    # 删除一些列，让数据更简单些，再去做后面的操作
    data = data.drop(["ma5", "ma10", "ma20", "v_ma5", "v_ma10", "v_ma20"], axis=1)
    print(data.head())
    print("--------------------------------")
    # 1 索引操作
    # 1.1直接使用行列索引(必须先列后行)
    print(data['open']['2018-02-27'])
    # print(data['2018-02-27']['open'])  # 会报错,不能先行后列
    # print(data[:, :])  # 不能切片取出

    # 1.2结合loc或者iloc属性使用索引
    # loc 根据索引值切片
    print(data.loc["2018-02-27":"2018-02-23", "open":"close"])
    # iloc 根据索引下标切片
    print(data.iloc[:5, :3])
    print("--------------------------------")
    # 1.3 使用ix组合索引(既可以通过下标也可以通过索引值获取)  此属性新版本会删除
    print(data.ix[0:4, ["open", "close"]])
    # 上一行jupyter 会警告提示 推荐使用loc和iloc来获取
    print(data.loc[data.index[0:4], ["open", "close"]])
    print(data.columns.get_indexer(["open", "close"]))  # 参数为列表,获取索引值对应的下标
    print(data.iloc[0:4, data.columns.get_indexer(["open", "close"])])


if __name__ == '__main__':
    main()
