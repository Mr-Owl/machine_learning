import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # 1 获取数据
    data = pd.read_csv("./data/nba_2017_nba_players_with_salary.csv")
    print(data.head())
    print(data.shape)
    # 查看数据综合描述 数量,最大最小,平均数,标准差,四分位
    print(data.describe())
    # 2 数据分析
    # 2.1 效率值相关性分析
    data_cor = data.loc[:, ['RPM', 'AGE', 'SALARY_MILLIONS', 'ORB', 'DRB',
                            'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                            'POINTS', 'GP', 'MPG', 'ORPM', 'DRPM']]
    print(data_cor)
    corr = data_cor.corr()
    # 获取量列数据之间的相关性 热力图
    # 热力图在实际中常用于展示一组变量的相关系数矩阵，
    # 在展示列联表的数据分布上也有较大的用途，
    # 通过热力图我们可以非常直观地感受到数值大小的差异状况。
    plt.figure(figsize=(20, 8), dpi=100)
    sns.heatmap(corr, square=True, linewidths=0.1, annot=True)
    plt.show()
