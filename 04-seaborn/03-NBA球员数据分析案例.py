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
    # 2.1 相关性分析
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
    # RPM为效率值.反映球员在场时对球队比赛获胜的贡献大小,最能反映球员的综合实力
    # 可以看RPM这一列与其他行的关系,看贡献和哪个关系最大
    sns.heatmap(corr, square=True, linewidths=0.1, annot=True)
    plt.show()
    # 3.2 球员数据分析
    # 按照效率值排名  效率值反映球员在场时对球队比赛获胜的贡献大小,最能反映球员的综合实力
    print(data.loc[:, ['PLAYER', 'RPM', 'AGE']].sort_values(by='RPM', ascending=False).head())
    # 按照效率值排名  效率值反映球员在场时对球队比赛获胜的贡献大小,最能反映球员的综合实力
    print(data.loc[:, ['PLAYER', 'RPM', 'AGE', 'SALARY_MILLIONS']].sort_values(by='SALARY_MILLIONS',
                                                                               ascending=False).head())
    # 利用seaborn中的distplot绘图看一下球员薪水,效率值,年龄这三个信息的分布情况
    # 分布及核密度展示
    plt.figure(figsize=(10, 10))
    sns.set_style('darkgrid')  # 设置绘图风格为核密度风格
    # plt.subplot(3, 1, 1)  # 表示将整个图像窗口分为3行1列,当前位置为1
    sns.displot(data['SALARY_MILLIONS'], kde=True)
    plt.ylabel('salary')
    plt.show()
    # plt.subplot(3, 1, 2)  # 表示将整个图像窗口分为3行1列,当前位置为2
    sns.displot(data['RPM'], kde=True)
    plt.ylabel('RPM')
    plt.show()
    # plt.subplot(3, 1, 3)  # 表示将整个图像窗口分为3行1列,当前位置为3
    sns.displot(data['AGE'], kde=True)
    plt.ylabel('AGE')
    plt.show()
