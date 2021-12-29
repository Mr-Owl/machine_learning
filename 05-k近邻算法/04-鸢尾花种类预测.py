from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

"""
KNN算法总结：
优点：
①简单有效
②重新训练的代价低
③适合类域交叉样本
KNN方法主要靠周围有限的邻近的样本,而不是靠判别类域的方法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，KNN方法较其他方法更为适合。
④适合大样本自动分类
该算法比较适用于样本容量比较大的类域的自动分类，而那些样本容量较小的类域采用这种算法比较容易产生误分。
缺点：
①惰性学习
KNN算法是懒散学习方法（lazy learning,基本上不学习），一些积极学习的算法要快很多
②类别评分不是规格化
不像一些通过概率评分的分类
③输出可解释性不强
例如决策树的输出可解释性就较强
④对不均衡的样本不擅长
当样本不平衡时，如一个类的样本容量很大，而其他类样本容量很小时，有可能导致当输入一个新样本时，该样本的K个邻居中大容量类的样本占多数。该算法只计算“最近的”邻居样本，某一类的样本数量很大，那么或者这类样本并不接近目标样本，或者这类样本很靠近目标样本。无论怎样，数量并不能影响运行结果。可以采用权值的方法（和该样本距离小的邻居权值大）来改进。
⑤计算量较大
目前常用的解决方法是事先对已知样本点进行剪辑，事先去除对分类作用不大的样本。
"""


def main():
    # 1 获取数据
    iris = load_iris()
    # 2 数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    # 3 特征工程 - 特征预处理(无量纲化)
    # 3.1 实例化对象转换器对象
    transfer = StandardScaler()
    # 3.2 转换
    x_train = transfer.fit_transform(x_train)
    # x_test = transfer.fit_transform(x_test)
    # transform使用训练集中的标准差进行表转化(因为测试集和训练集采用的是同一个数据集,正太分布一致,所以标准差相同)
    # 使用transform前需使用fit_transform 有一个已经计算好的标准差
    x_test = transfer.transform(x_test)
    # 4 机器学习-knn
    # 4.1 实例化估计器
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 4.2 模型训练
    estimator.fit(x_train, y_train)
    # 5 模型评估
    # 5.1预测结果输出
    y_pre = estimator.predict(x_test)
    print("预测值是:\n", y_pre)
    print("预测值和真实值的对比时是:\n", y_pre == y_test)
    score = estimator.score(x_test, y_test)
    print("准确率为:", score)


if __name__ == '__main__':
    main()
