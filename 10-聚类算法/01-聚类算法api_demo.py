import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

"""
sklearn.cluster.KMeans(n_clusters=8)
参数:
n_clusters:开始的聚类中心数量
整型，缺省值=8，生成的聚类数，即产生的质心（centroids）数。
方法:
estimator.fit(x)
estimator.predict(x)
estimator.fit_predict(x)
计算聚类中心并预测每个样本属于哪个类别,相当于先调用fit(x),然后再调用predict(x
"""


def main():
    """
    聚类算法api
    :return:
    """
    # 创造数据集
    X, y = make_blobs(n_samples=1000,  # 样本数
                      n_features=2,  # 特征数
                      centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],  # 中心点
                      cluster_std=[0.4, 0.2, 0.2, 0.2],  # 方差，数据的离散程度
                      random_state=9)
    print(type(X))
    # plt.scatter(X[:, 0], X[:, 1], marker="o")
    # plt.show()
    # kmeans训练，且可视化
    y_pre = KMeans(n_clusters=4, random_state=9).fit_predict(X)
    # 可视化展示
    plt.scatter(X[:, 0], X[:, 1], c=y_pre)
    plt.show()
    # ch_scole 查看最后效果
    print(calinski_harabasz_score(X, y_pre))


if __name__ == '__main__':
    main()
