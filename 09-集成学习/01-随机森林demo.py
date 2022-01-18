import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

"""
随机森林api:
sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, 
max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)

n_estimators：integer，optional(default = 10)森林里的树木数量120,200,300,500,800,1200
在利用最大投票数或平均值来预测之前，你想要建立子树的数量。

Criterion：string，可选(default =“gini”)
分割特征的测量方法

max_depth：integer或None，可选(默认=无)
树的最大深度 5,8,15,25,30

max_features="auto”,每个决策树的最大特征数量
If "auto", then max_features=sqrt(n_features). 
If "sqrt", then max_features=sqrt(n_features)(same as "auto").
If "log2", then max_features=log2(n_features).
If None, then max_features=n_features.

bootstrap：boolean，optional(default = True)
是否在构建树时使用放回抽样

min_samples_split 内部节点再划分所需最小样本数
这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分，默认是2。
如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

min_samples_leaf 叶子节点的最小样本数
这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝， 默认是1。
叶是决策树的末端节点。 较小的叶子使模型更容易捕捉训练数据中的噪声。
一般来说，我更偏向于将最小叶子节点数目设置为大于50。

min_impurity_split: 节点划分最小不纯度
这个值限制了决策树的增长，如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。
一般不推荐改动默认值1e-7。

上面决策树参数中最重要的包括：
①最大特征数max_features，
②最大深度max_depth，
③内部节点再划分所需最小样本数min_samples_split
④叶子节点最少样本数min_samples_leaf。
"""
def main():
    """
    随机森林api的简单实用
    :return:
    """
    # 1 获取数据
    titan = pd.read_csv("./data/titanic.csv")
    # 2 数据基本处理
    # 2.1 确认特征值，目标值
    x = titan[["pclass", "age", "sex"]]
    y = titan["survived"]
    # 2.2 缺失值处理
    x["age"].fillna(value=titan["age"].mean(), inplace=True)
    # 2.3 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
    # 3 特征提取（字典特征抽取）
    x_train = x_train.to_dict(orient="records")
    x_test = x_test.to_dict(orient="records")
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4 机器学习（决策树）
    # 4.1 实例化
    rf = RandomForestClassifier()
    # 4.2 通过超参数调优，训练模型
    param = {"n_estimators": [100, 120, 300], "max_depth": [3, 7, 11]}
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    # 5 模模型评估
    gc.fit(x_train, y_train)
    print("随机森林预测结果是：\n", gc.score(x_test, y_test))


if __name__ == '__main__':
    main()
