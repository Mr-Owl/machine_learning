import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    movie = pd.read_csv("./data/IMDB-Movie-Data.csv")
    print(movie)
    print("-----------------------------")
    # 1 缺失值处理 为nan
    # 1.1 pd.is/notnull(数据)
    print(np.all(pd.notnull(movie)))  # 里面如果有一个缺失值,那么返回False,说明有缺失值
    print(np.any(pd.isnull(movie)))  # 里面如果有一个缺失值,那么就返回True,说明有缺失值
    print("-----------------------------")
    # 1.2 dopna() 删除缺失值
    data = movie.dropna()
    print(np.all(pd.notnull(data)))
    print("-----------------------------")
    # 1.3 填充缺失值 fillna(替换值, inplace=True是否修改原对象)
    # 将缺失值修改为中位数
    # movie["Revenue (Millions)"].fillna(movie["Revenue (Millions)"].mean(), inplace=True)
    # print(movie)
    for i in movie.columns:
        if np.any(pd.isnull(movie[i])):
            print(i)
            movie[i].fillna(movie[i].mean(), inplace=True)
    print("----------------------------")
    # 缺失值处理不为nan 为其他值
    # replace(to_replace="默认值", value=np.nan)
    # to_replace 需要替换的内容  value替换后的内容
    wis = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
    wis = wis.replace(to_replace="?", value=np.nan)
    # print(wis)
    print(np.any(pd.isnull(wis)))
