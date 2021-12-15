import pandas as pd
import matplotlib.pyplot as plt


def main():
    # 读取csv数据
    data = pd.read_csv("./data/stock_day.csv")
    print(data)
    # 1 算术运算
    print(data["open"].add(1).head())  # 加
    # print(data["open"]+10)  # 一般不用
    print(data["open"].sub(1).head())  # 减
    print("----------------------")
    # 2 逻辑运算
    # 2.1 逻辑运算
    print(data["open"] > 20)  # 根据条件将数据改为True/False
    print(data[data["open"] > 23].head())  # 筛选出符合条件的数据
    print((data["open"] > 23) & (data["open"] < 24))
    print(data[(data["open"] > 23) & (data["open"] < 24)].head()["open"])
    print("----------------------")
    # 2.2 逻辑运算函数
    # 对象.query(查询字符串)
    print(data.query("open>23 & open<24").head()["open"])
    # 对象["列名"].isin(values) 查看前面对象中的列值是否在values里面 在True 不在False
    print(data[data["open"].isin([23.23, 24.88])].head()["open"])
    print("----------------------")
    # 3 统计函数
    # 3.1 describe
    # 综合分析: 能够直接得出很多统计结果,count, mean, std, min, max 等
    # 会分成四份 25% 50% 75% 对应的值  四分位   三个切割点的值
    # 计算平均值、标准差、最大值、最小值
    print(data.describe())
    print("----------------------")
    # max min 最大值 最小值
    # 使用统计函数：0 代表列求结果， 1 代表行求统计结果
    print(data.max(0))
    # median 中位数
    print(data.median(0))
    # std()、var()  方差 标准差
    # idxmax  idxmin 索引最大值  索引最小值
    print(data.idxmax())
    print(data.idxmin())
    print("----------------------")
    # 3.2 累计统计函数
    data = data.sort_index()
    print(data.head())
    stock_rise = data["p_change"]
    print("----------------------")
    print(stock_rise.cumsum())
    stock_rise.cumsum().plot()
    print("----------------------")
    # plt.show()
    # 4 自定义运算
    # apply(func, axis=0)
    # func:自定义函数
    # axis=0:默认是列，axis=1为行进行运算
    print(data[["open","close"]].apply(lambda x:x.max()-x.min(), axis=0))

if __name__ == '__main__':
    main()
