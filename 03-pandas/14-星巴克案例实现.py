import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 数据来源: https://www.kaggle.com/damianpanek/sunday-eda/data
    starbucks = pd.read_csv("./data/starbucks/directory.csv")
    # print(starbucks)
    # 分组
    # print(starbucks.groupby(["Country"]).count())
    # 按国家分组,查看每个国家开得店铺数量
    count = starbucks.groupby(["Country"]).count()
    count["Brand"].plot(kind="bar", figsize=(20, 8))
    # plt.show()
    # 按国家省份分组,查看每个国家各个省份的店铺数量
    print(starbucks.groupby(["Country", "State/Province"]).count())