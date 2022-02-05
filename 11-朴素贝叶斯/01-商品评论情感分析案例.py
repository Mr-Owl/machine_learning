import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

"""
sklearn.naive_bayes.MultinomialNB(alpha = 1.0)
朴素贝叶斯分类
alpha：拉普拉斯平滑系数
"""


def main():
    # 1 获取数据
    data = pd.read_csv("./data/书籍评价.csv", encoding="gbk")
    # print(data)
    # 2 数据基本处理
    # 2.1 取出内容列，用于后面分析
    content = data["内容"]
    # print(content)
    # 2.2把评价中的好评差评转换为数字
    data.loc[data.loc[:, "评价"] == "好评", "评论编号"] = 1
    data.loc[data.loc[:, "评价"] == "差评", "评论编号"] = 0
    # 2.3选择停用词
    stopwords = []
    with open("./data/stopwords.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for tmp in lines:
            line = tmp.strip()
            stopwords.append(line)
    # 去重
    stopwords = list(set(stopwords))
    # 2.4 把"内容"成绩,转化标准格式
    comment_list = []
    for tmp in content:
        # 把每句话转换成一个个词
        # cut_all 参数默认为 False,所有使用 cut 方法时默认为精确模式
        seg_list = jieba.cut(tmp, cut_all=False)
        # print(seg_list)  # 该返回值为一个对象
        # 转换成字符串
        seg_str = ",".join(seg_list)
        comment_list.append(seg_str)
    # 2.5 统计词的个数
    con = CountVectorizer(stop_words=stopwords)
    X = con.fit_transform(comment_list)
    # print(X.toarray())
    # print(con.get_feature_names())
    ## 2.6准备训练集和测试集
    x_train = X.toarray()[:10, :]
    y_train = data["评价"][:10]
    x_test = X.toarray()[10:, :]
    y_test = data["评价"][10:]
    # 3 模型训练
    mb = MultinomialNB(alpha=1)
    mb.fit(x_train, y_train)
    y_pre = mb.predict(x_test)
    print("预测值：", y_pre)
    print("真实值：", y_test)
    # 模型评估
    print("准确率：", mb.score(x_test, y_test))



if __name__ == '__main__':
    main()
