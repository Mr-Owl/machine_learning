import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

"""
特征降维：
降维是指在某些限定条件下，降低随机变量(特征)个数，得到一组“不相关”主变量的过程
①降低随机变量的个数
②相关特征(correlated feature)
    相对湿度与降雨量之间的相关等等
"""


def var_thr():
    """
    特征选择：低方法特征过滤
    :return:
    """
    data = pd.read_csv("./data/factor_returns.csv")
    # 把小于threshold的特征过滤掉
    transfer = VarianceThreshold(threshold=1)
    transfer_data = transfer.fit_transform(data.iloc[:, 1:10])
    print("过滤前特征：\n", data.iloc[:, 1:10].shape)
    print("过滤后特征：\n", transfer_data.shape)


def pea_demo():
    """
    相关系数法：皮尔逊相关系数
    :return:
    """
    # 准备数据
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    ret = pearsonr(x1, x2)
    print("皮尔逊相关系数的结果是：\n", ret)


def spea_demo():
    """
    相关系数法：斯皮尔曼相关系数
    :return:
    """
    # 准备数据
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    ret = spearmanr(x1, x2)
    print("斯皮尔曼相关系数的结果是：\n", ret)


def pca_demo():
    """
    主成分分析：pca降维
    :return:
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # pca 小数保留百分比
    transfer = PCA(n_components=0.9)
    trans_data = transfer.fit_transform(data)
    print("保留0.9的数据最后维度为：\n", trans_data)
    # pca 整数保留维度信息
    transfer = PCA(n_components=3)
    trans_data = transfer.fit_transform(data)
    print("保留3列：\n", trans_data)

if __name__ == '__main__':
    # var_thr()
    # pea_demo()
    # spea_demo()
    pca_demo()