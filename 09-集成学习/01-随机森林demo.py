import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
    gc.fit(x_train, y_train)
    print("随机森林预测结果是：\n", gc.score(x_test, y_test))


if __name__ == '__main__':
    main()
