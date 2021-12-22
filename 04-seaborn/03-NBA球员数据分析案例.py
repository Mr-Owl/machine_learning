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
    # 2.2 球员数据分析
    # 按照效率值排名  效率值反映球员在场时对球队比赛获胜的贡献大小,最能反映球员的综合实力
    print(data.loc[:, ['PLAYER', 'RPM', 'AGE']].sort_values(by='RPM', ascending=False).head())
    # 按照效率值排名  效率值反映球员在场时对球队比赛获胜的贡献大小,最能反映球员的综合实力
    print(data.loc[:, ['PLAYER', 'RPM', 'AGE', 'SALARY_MILLIONS']].sort_values(by='SALARY_MILLIONS',
                                                                               ascending=False).head())
    # 利用seaborn中的distplot绘图看一下球员薪水,效率值,年龄这三个信息的分布情况
    # 单变量分析
    # 2.3分布及核密度展示
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
    # 2.4双变量分析
    sns.jointplot(data.AGE, data.SALARY_MILLIONS, kind="hex")
    plt.show()
    # 2.3多变量
    mulit_data = data.loc[:, ['RPM', 'SALARY_MILLIONS', 'AGE', 'POINTS']]
    sns.pairplot(mulit_data)
    plt.show()


    # 2.4衍生变量的一些可视化实践-以年龄为例
    # 查看不同年龄段每分钟得分情况
    def age_cut(df):
        if df.AGE <= 24:
            return "young"
        elif df.AGE >= 30:
            return "old"
        else:
            return "best"


    # 使用apply对年龄进行划分
    data['age_cut'] = data.apply(lambda x: age_cut(x), axis=1)
    print(data.loc[:, ['PLAYER', 'age_cut']])
    # 方便计数
    data['cut'] = 1

    print(data.loc[data.age_cut == "best"].SALARY_MILLIONS.head())
    # 基于年龄对球员薪水和效率值进行分析
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 10), dpi=100)
    plt.title("RPM and SALARY")
    x1 = data.loc[data.age_cut == "old"].SALARY_MILLIONS
    y1 = data.loc[data.age_cut == "old"].RPM
    plt.plot(x1, y1, "^")

    x2 = data.loc[data.age_cut == "best"].SALARY_MILLIONS
    y2 = data.loc[data.age_cut == "best"].RPM
    plt.plot(x2, y2, "^")

    x3 = data.loc[data.age_cut == "young"].SALARY_MILLIONS
    y3 = data.loc[data.age_cut == "young"].RPM
    plt.plot(x3, y3, ".")
    plt.show()
    # 老中青三代计数统计分布情况
    multi_data2 = data.loc[:, ['RPM', 'POINTS', 'TRB', 'AST', 'STL', 'BLK', 'age_cut']]
    sns.pairplot(multi_data2, hue="age_cut")
    plt.show()
    # 3球队数据分析
    # 3.1 球队薪资排行
    # 按老青幼分组,看薪水平均值/最大值/最小值等
    print(data.groupby(by="age_cut").agg({"SALARY_MILLIONS": np.mean}))  # agg聚合条件
    print("--------------------------")
    # 按球队分组,查看每个球队的薪资情况
    data_team = data.groupby(by="TEAM").agg({"SALARY_MILLIONS": np.mean})
    print(data_team.head())
    # 查看球队薪资排名
    print(data_team.sort_values(by="SALARY_MILLIONS", ascending=False).head(10))
    # 按照分球队分年龄段,按上榜球员降序排列,上榜球员数相同,则按效率值降序排名
    data_rpm = data.groupby(by=["TEAM", "age_cut"]).agg(
        {"SALARY_MILLIONS": np.mean, "RPM": np.mean, "PLAYER": np.size})
    print(data_rpm.sort_values(by=["PLAYER", "RPM"], ascending=False).head(10))
    # 按照球队综合实力排名,
    data_rpm2 = data.groupby(by=["TEAM"], as_index=False).agg(
        {"SALARY_MILLIONS": np.mean, "RPM": np.mean, "PLAYER": np.size, "POINTS": np.mean, "eFG%": np.mean,
         "MPG": np.mean, "AGE": np.mean})
    print(data_rpm2.sort_values(by="RPM", ascending=False).head(10))
    # 箱形图和小提琴图进行数据分析
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 10))
    # 获取所需的队伍数据
    data_team2 = data[data.TEAM.isin(["GS", "CLE", "SA", "LAC", "OKC", "UHAT", "CHA", "TOR", "NO", "BOS"])]
    # plt.subplot(行,列,索引)
    # 绘制箱形图
    sns.boxplot(x="TEAM", y="SALARY_MILLIONS", data=data_team2)
    plt.show()
    sns.boxplot(x="TEAM", y="AGE", data=data_team2)
    plt.show()
    sns.boxplot(x="TEAM", y="MPG", data=data_team2)
    plt.show()
    # 绘制小提琴图
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 10))
    # 3分球命中率分析
    sns.violinplot(x="TEAM", y="3P%", data=data_team2)
    plt.show()
    # 真实命中率分析
    sns.violinplot(x="TEAM", y="eFG%", data=data_team2)
    plt.show()
    # 得分情况分析
    sns.violinplot(x="TEAM", y="POINTS", data=data_team2)
    plt.show()
