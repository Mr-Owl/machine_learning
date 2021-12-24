import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import mpl

# 字体需要额外下载安装，另外配置后需要去家目录cache清除缓存
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

if __name__ == '__main__':
    # 1获取数据
    file_data = pd.read_csv("./data/链家北京租房数据.csv")
    print(file_data.head())
    print(file_data.shape)
    # 获取基本信息
    print(file_data.info())
    print(file_data.describe())
    # ２数据基本处理
    # ２.１重复值　空值检测与删除
    # 　重复值检测
    print(file_data.duplicated())
    #   重复值删除
    file_data = file_data.drop_duplicates()
    print(file_data.shape)
    #   空值删除
    if np.any(pd.isnull(file_data)):
        file_data = file_data.dropna()
        print("删除空值")
    print(file_data.shape)
    # 2.2 数据类型转换
    # 面积数据类型转换
    # 取出面积中的数字部分
    data_new = np.array([])
    data_area = file_data["面积(㎡)"].values
    for i in data_area:
        data_new = np.append(data_new, np.array(i[:-2]))
    # 转换数据类型
    data_new = data_new.astype(np.float64)
    # 替换面积数据
    file_data.loc[:, "面积(㎡)"] = data_new
    print(file_data.head())
    # 户型表达方式替换
    house_data = file_data["户型"]
    temp_list = []
    for i in house_data:
        new_info = i.replace("房间", "室")
        temp_list.append(new_info)
    file_data.loc[:, "户型"] = temp_list
    print(file_data)
    # 3 图表分析
    # 房源数量,位置分布分析
    new_df = pd.DataFrame({"区域": file_data["区域"].unique(), "数量": [0] * 13})
    # 获取每个区域房源数量
    area_count = file_data.groupby(by="区域").count()
    new_df["数量"] = area_count.values
    print(new_df)
    # 户型数量分析
    house_data = file_data["户型"]
    print(house_data.head())


    def all_houese(arr):
        key = np.unique(arr)
        result = {}
        for k in key:
            mask = (arr == k)
            arr_new = arr[mask]

            v = arr_new.size
            result[k] = v
        return result


    house_info = all_houese(house_data)
    # 去掉统计数量较少的值
    house_data = dict((k, v) for k, v in house_info.items() if v > 50)
    show_house = pd.DataFrame({"户型": [x for x in house_data.keys()],
                               "数量": [x for x in house_data.values()]})
    print(show_house)
    # 图形展示房屋类型
    house_type = show_house["户型"]
    house_type_num = show_house["数量"]
    plt.barh(range(11), house_type_num)
    plt.yticks(range(11), house_type)
    plt.xlim(0, 2500)  # 设置x轴数据范围
    plt.title("北京市各区域租房数量统计")
    plt.xlabel("数量")
    plt.ylabel("房屋类型")
    for x, y in enumerate(house_type_num):
        plt.text(y + 0.5, x - 0.2, "%s" % y)
    plt.show()
    # 平均租金分析
    df_all = pd.DataFrame({"区域": file_data["区域"].unique(),
                           "房屋总金额": [0] * 13,
                           "总面积": [0] * 13})
    sum_price = file_data["价格(元/月)"].groupby(file_data["区域"]).sum()
    sum_area = file_data["面积(㎡)"].groupby(file_data["区域"]).sum()
    df_all["房屋总金额"] = sum_price.values
    df_all["总面积"] = sum_area.values
    # 计算各个区域每平方米的房子
    df_all["每平方米租金(元)"] = round(df_all["房屋总金额"] / df_all["总面积"], 2)
    print(df_all)
    df_merge = pd.merge(new_df, df_all)
    print(df_merge)