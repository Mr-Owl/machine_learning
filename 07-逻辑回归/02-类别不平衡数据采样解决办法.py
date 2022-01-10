from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def main():
    # 使用make_classification生成样本数据
    X, y = make_classification(n_samples=5000,
                               n_features=2,  # 特征个数= n_informative（） + n_redundant + n_repeated
                               n_informative=2,  # 多信息特征的个数
                               n_redundant=0,  # 冗余信息，informative特征的随机线性组合
                               n_repeated=0,  # 重复信息，随机提取n_informative和n_redundant 特征
                               n_classes=3,  # 分类类别
                               n_clusters_per_class=1,  # 某一个类别是由几个cluster构成的
                               weights=[0.01, 0.05, 0.94],  # 列表类型，权重比
                               random_state=0)

    # 查看各个标签的样本量
    print(Counter(y))  # Counter({2: 4674, 1: 262, 0: 64})
    # 数据集可视化
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    """
    目前样本数据类别严重不平衡
    
    关于类别不平衡的问题，主要有两种处理方式：
    过采样方法
        增加数量较少那一类样本的数量，使得正负样本比例均衡。
    欠采样方法
        减少数量较多那一类样本的数量，使得正负样本比例均衡。
    """
    # 1 过采样
    # 对训练集里的少数类进行“过采样”（oversampling），
    # 即增加一些少数类样本使得正、反例数目接近，然后再进行学习。
    # 1.1 随机过采样
    # 在少数类的样本中随机选择一些样本，然后复制所选择的样本添加进去来扩大少数类的样本
    # 使用imblearn进行随机过采样
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    # 查看结果
    print(Counter(y_resampled))
    # 过采样后样本结果
    # Counter({2: 4674, 1: 4674, 0: 4674})
    # 数据集可视化
    plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
    plt.show()
    """
    缺点：
        对于随机过采样，由于需要对少数类样本进行复制来扩大数据集，造成模型训练复杂度加大。
        另一方面也容易造成模型的过拟合问题，因为随机过采样是简单的对初始样本进行复制采样，这就使得学习器学得的规则过于具体化，不利于学习器的泛化性能，造成过拟合问题。
        为了解决随机过采样中造成模型过拟合问题，又能保证实现数据集均衡的目的，出现了过采样法代表性的算法SMOTE算法。
    """
    # 1.2 过采样代表性算法-SMOTE
    """
    SMOTE全称是Synthetic Minority Oversampling即合成少数类过采样技术。
    SMOTE算法合成新少数类样本的算法描述如下：

    1) 对于少数类中的每一个样本xi ，以欧氏距离为标准计算它到少数类样本集 Smin 中所有样本的距离，得到其k近邻。
    2) 根据样本不平衡比例设置一个采样比例以确定采样倍率N，对于每一个少数类样本 xi ，从其k近邻中随机选择若干个样本，假设选择的是 x'i 。
    3) 对于每一个随机选出来的近邻 x'i ，分别与 xi按照如下公式构建新的样本。
    xnew = xi + rand(0,1)*(x'i-xi)
    SMOTE算法摒弃了随机过采样复制样本的做法，可以防止随机过采样中容易过拟合的问题，实践证明此方法可以提高分类器的性能。
    """
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    Counter(y_resampled)
    # 采样后样本结果
    # [(0, 4674), (1, 4674), (2, 4674)]
    # 数据集可视化
    plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
    plt.show()
    # 2 欠采样
    # 直接对训练集中多数类样本进行“欠采样”（undersampling），即去除一些多数类中的样本
    # 使得正例、反例数目接近，然后再进行学习。
    # 2.1 随机欠采样
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(Counter(y_resampled))
    plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
    plt.show()
    """
    缺点：
    随机欠采样方法通过改变多数类样本比例以达到修改样本分布的目的，
    从而使样本分布较为均衡，但是这也存在一些问题。对于随机欠采样，
    由于采样的样本集合要少于原来的样本集合，因此会造成一些信息缺失，
    即将多数类样本删除有可能会导致分类器丢失有关多数类的重要信息。
"""



if __name__ == '__main__':
    main()
