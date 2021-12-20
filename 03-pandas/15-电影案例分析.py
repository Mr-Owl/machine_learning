# %matplotlib inline 该方法为ipython中的魔法方法,用来防止在某些环境下matplotlib显示不出来的问题
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 案例来源 :https://www.kaggle.com/damianpanek/sunday-eda/data
    # 文件的路径
    path = "./data/IMDB-Movie-Data.csv"
    # 读取文件
    movie = pd.read_csv(path)
    # 获取所有电影评分的平均分
    print(movie["Rating"].mean())
    # 获取所有导演,需要去重
    # 使用np去重
    print(np.unique(movie["Director"]).shape[0])
    # 使用DataFrame去重
    print(movie["Director"].unique().shape[0])
    # Rating 分布 直方图
    # 第一种:直接绘制
    # movie["Rating"].plot(kind ="hist")  # 刻度和直方图不对齐
    # plt.show()
    # 第二种:使用matplotlib绘制
    # 1 创建画布
    plt.figure(figsize=(20, 8), dpi=100)
    # 2 绘制图像 直方图,分20组 此时还是不对齐
    plt.hist(movie["Rating"].values, bins=20)
    # 3 修改刻度使之对齐  将数据根据最大最小值,等间距分20组,所以就需要21个刻度,比如2组就是3个刻度
    max_rating = movie["Rating"].max()
    min_rating = movie["Rating"].min()
    # np.linspace 根据最小值 最大值 返回要求数量的数字的数组
    ticks = np.linspace(min_rating, max_rating, num=21)
    # 修改刻度
    plt.xticks(ticks=ticks)
    # 添加网格
    plt.grid()
    plt.show()
    # 统计电影分类(genre)的情况
    # 1、创建一个全为0的dataframe，列索引置为电影的分类，temp_df
    # 2、遍历每一部电影，temp_df中把分类出现的列的值置为1
    # 3、求和
    # 1 获取所有电影分类
    temp_list = [i.split(",") for i in movie["Genre"]]
    # 2 去重temp_list[i]
    genre_list = np.unique([i for j in temp_list for i in j])
    # 3 生成全为0的DataFrame,列索引设置为电影的分类
    temp_movie = pd.DataFrame(np.zeros([movie.shape[0], genre_list.shape[0]]), columns=genre_list)
    # print(temp_movie)
    # 4 修改每一个电影在temp_movie中的分类
    for i in range(1000):
        # temp_movie.ix[i, temp_list[i]] 混合索引新版本不支持
        # 设置行索引为i,列索引为temp_list[i]的所有值为1
        temp_movie.loc[i, temp_list[i]] = 1
    # print(temp_movie)
    genre = temp_movie.sum().sort_values(ascending=False)
    print(genre)
    # 5 绘制图形  colormap设置绘图风格
    genre.plot(kind="bar", colormap="cool", figsize=(20, 9), fontsize=16)
    plt.show()
