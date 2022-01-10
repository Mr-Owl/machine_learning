"""
1.获取数据
2.基本数据处理
2.1 缺失值处理
2.2 确定特征值,目标值
2.3 分割数据
3.特征工程(标准化)
4.机器学习(逻辑回归)
5.模型评估
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 关闭ssl验证
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

"""
sklearn.linear_model.LogisticRegression(solver='liblinear', 
penalty=‘l2’, C = 1.0)

solver可选参数:{'liblinear', 'sag', 'saga','newton-cg', 'lbfgs'}，
默认: 'liblinear'；用于优化问题的算法。
对于小数据集来说，“liblinear”是个不错的选择，而“sag”和'saga'对于大型数据集会更快。
对于多类问题，只有'newton-cg'， 'sag'， 'saga'和'lbfgs'可以处理多项损失;
“liblinear”仅限于“one-versus-rest”分类。

penalty：正则化的种类
C：正则化力度

默认将类别数量少的当做正例
LogisticRegression方法相当于 SGDClassifier(loss="log", penalty=" "),
SGDClassifier实现了一个普通的随机梯度下降学习。
而使用LogisticRegression(实现了SAG)
"""


def main():
    """
    癌症分类预测-良／恶性乳腺癌肿瘤预测
    :return:
    """
    # 原始数据的下载地址：https://archive.ics.uci.edu/ml/machine-learning-databases/
    names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
             'Normal Nucleoli', 'Mitoses', 'Class']
    # data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=names)
    # data.to_csv("./data/test.csv")
    # 1 获取数据
    data = pd.read_csv("./data/test.csv")
    # print(data.head())
    # 2 基本数据处理
    # 2.1 缺失值处理
    data = data.replace(to_replace="?", value=np.nan)
    data = data.dropna()
    # 2.2 确认特征值，目标值
    x = data.iloc[:, 1:-1]
    # print(x.head())
    y = data["Class"]
    # 2.3 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)
    # 3 特征工程（标准化）
    # 实例化标准化对象
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4 机器学习（逻辑回归）
    # 实例化估计器
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    # 5 模型评估
    # 5.1 准确率
    ret = estimator.score(x_test, y_test)
    print("准确率为：", ret)
    print("----------------------------------")
    # 5.2 预测值
    y_pre = estimator.predict(x_test)
    print("预测值:\n", y_pre)
    print("真实值：\n", y_test)
    print("----------------------------------")
    # 5.3 精确率\召回率指标评价
    ret = classification_report(y_test, y_pre, labels=(2, 4), target_names=("良性", "恶性"))
    print(ret)
    print("----------------------------------")


if __name__ == '__main__':
    main()
