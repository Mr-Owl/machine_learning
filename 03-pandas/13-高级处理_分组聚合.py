import pandas as pd

if __name__ == '__main__':
    col = pd.DataFrame(
        {'color': ['white', 'red', 'green', 'red', 'green'], 'object': ['pen', 'pencil', 'pencil', 'ashtray', 'pen'],
         'price1': [5.56, 4.20, 1.30, 0.56, 2.75], 'price2': [4.75, 4.12, 1.60, 0.75, 3.15]})
    # print(col)
    # 分组与聚合通常是分析数据的一种方式，通常与一些统计函数一起使用，查看数据的分组情况
    # DataFrame.groupby(key, as_index=False)
    # key:分组的列数据，可以多个
    # as_index 默认为True保留原有的索引,Fakse不保留原有的索引
    print(col.groupby(["color"])["price1"].mean())
    print(col.groupby(["color"], as_index=False)["price1"].mean())
    print("---------------------------------------------")
    print(col["price1"].groupby(col["color"]).mean())

