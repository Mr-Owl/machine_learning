import numpy as np
import matplotlib.pyplot as plt


def main():
    # # 生成正态分布的数组 np.random.normal(loc=均值，scale=标准差，size=样本维度)
    # x1 = np.random.normal(1.75, 1, 100000000)
    # print(x1)
    # # 画图看分布状况
    # # 1 创建画布
    # plt.figure(figsize=(20, 10), dpi=100)
    # # 2 绘制直方图  数据，分组数量
    # plt.hist(x1, 1000)
    # # 3 显示图像
    # plt.show()

    # # 创建符合标准正态分布的4只股票偶的五天的长滴润肤数据
    # stock_change = np.random.normal(0, 1, (4, 5))
    # print(stock_change)

    # 均匀分布
    x2 = np.random.uniform(-1, 1, 100000000)
    # 画图看分布状况
    # 1 创建画布
    plt.figure(figsize=(20, 10), dpi=100)
    # 2 绘制直方图  数据，分组数量
    plt.hist(x2, 1000)
    # 3 显示图像
    plt.show()



if __name__ == '__main__':
    main()
