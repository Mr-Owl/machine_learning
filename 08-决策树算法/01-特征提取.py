from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

"""
特征提取
将任意数据（如文本或图像）转换为可用于机器学习的数字特征
特征值化是为了计算机更好的去理解数据

特征提取分类:
字典特征提取(特征离散化)
文本特征提取
图像特征提取（深度学习将介绍）

特征提取API
sklearn.feature_extraction


"""


def dict_demo():
    """
    字典特征提取
    作用：对字典数据进行特征值化
    sklearn.feature_extraction.DictVectorizer(sparse=True,…)
    DictVectorizer.fit_transform(X)
    X:字典或者包含字典的迭代器返回值
    返回sparse矩阵,值False返回普通的one_hot编码矩阵
    sparse矩阵相对普通one-hot编码矩阵更节省空间，效率更高
    DictVectorizer.get_feature_names() 返回类别名称
    :return:
    """
    # 1 获取数据
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 2 字典特征提取
    # 2.1 实例化
    # transfer = DictVectorizer(sparse=False)
    transfer = DictVectorizer(sparse=True)
    # 2.2 转换
    new_data = transfer.fit_transform(data)
    print(new_data)
    # 2.3 获取具体属性名
    print(transfer.get_feature_names())


def english_count_demo():
    """
    文本特征提取-英文
    作用：对文本数据进行特征值化
    api:
    sklearn.feature_extraction.text.CountVectorizer(stop_words=[])
    返回词频sparse矩阵
    不统计标点符号和单个字母
    stop_words 设置不参考的词语
    CountVectorizer.fit_transform(X)
    X:文本或者包含文本字符串的可迭代对象
    返回值:返回sparse矩阵
    CountVectorizer.get_feature_names() 返回值:单词列表
    :return:
    """
    # 获取数据
    data = ["life is short,i like python",
            "life is too long,i dislike python"]
    # 文本特征转换
    # transfer = CountVectorizer()  # 注意：没有sparse这个参数
    transfer = CountVectorizer(stop_words=["dislike"])
    new_data = transfer.fit_transform(data)

    # 查看特征名字
    names = transfer.get_feature_names()

    print("特征名字是：", names)
    print(new_data)
    print(new_data.toarray())  # 转换成one-hot矩阵


if __name__ == '__main__':
    # dict_demo()
    english_count_demo()
