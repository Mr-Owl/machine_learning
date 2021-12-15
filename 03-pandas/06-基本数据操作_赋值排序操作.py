import pandas as pd


def main():
    # 读取csv数据
    data = pd.read_csv("./data/stock_day.csv")
    print(data)
    # 1 赋值操作
    data['close'] = 1
    print(data.loc[:"2018-02-23", "close"])
    data.close = 10
    print(data.iloc[:3, data.columns.get_indexer(["close"])])
    print("---------------------------")
    # 2 排序   索引排序和值排序两种方式
    # 2.1DataFrame排序
    # 使用df.sort_values(by=, ascending=)
    # 单个键或者多个键进行排序,
    # 参数：
    # by：指定排序参考的键
    # ascending: 默认升序
    # ascending = False:降序
    # ascending = True:升序
    print(data.sort_values(by="open", ascending=True).iloc[:, :3])
    print(data.sort_values(by=["open", "high"]).iloc[:, :3])
    print(data.sort_index().head(3))  # 按索引排序
    print("----------------------")
    # 2.2 Series排序
    # 使用series.sort_values(ascending=True)进行排序
    # series排序时，只有一列，不需要参数
    print(data["high"].sort_values().head())
    print(data["high"].sort_index().head())  # 按索引排序


if __name__ == '__main__':
    main()
