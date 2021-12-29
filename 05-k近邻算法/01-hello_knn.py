# coding:utf-8

from sklearn.neighbors import KNeighborsClassifier

def main():
    """
    如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
    流程：
    1）计算已知类别数据集中的点与当前点之间的距离（欧氏距离）
    当前样本与已知的多个样本的对应特征差值平方的和
    2）按距离递增次序排序
    3）选取与当前点距离最小的k个点
    4）统计前k个点所在的类别出现的频率
    5）返回前k个点出现频率最高的类别作为当前点的预测分类
    :return:
    """
    # 1 构造数据
    x = [[1], [2], [10], [20]]  # 样本值 多行多列,d ataframe
    y = [0, 0, 1, 1]  # 目标值,一列,series
    # 2 训练模型
    # 2.1 实例化一个估计器对象
    estimator = KNeighborsClassifier(n_neighbors=1)
    # 2.2 调用fit方法,进行训练
    estimator.fit(x,y)
    # 3 模型预测
    ret = estimator.predict([[0]])
    print(ret)
    ret1 = estimator.predict([[110]])
    print(ret1)
if __name__ == '__main__':
    main()