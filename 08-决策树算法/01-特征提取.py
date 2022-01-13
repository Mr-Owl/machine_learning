import jieba
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
    不统计标点符号和单个字母，使用空格和符号进行切割
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


def chinese_count_demo():
    """
    文本特征提取-中文
    作用：对文本数据进行特征值化
    api:
    :return:
    """
    # 获取数据
    data = ["人生苦短，我喜欢Python",
            "生活太长久，我不喜欢Python"]
    # 文本特征转换
    transfer = CountVectorizer()
    new_data = transfer.fit_transform(data)

    # 查看特征名字
    names = transfer.get_feature_names()

    print("特征名字是：", names)
    print(new_data)
    print(new_data.toarray())  # 转换成one-hot矩阵


def cut_word(text):
    """
    对中文进行分词
    "我爱北京"  -----> "我 爱 北京"
    :param text:
    :return:text
    """
    text = " ".join(list(jieba.cut(text)))
    return text


def chinese_count_demo2():
    """
    文本特征提取-中文
    作用：对文本数据进行特征值化
    api:
    :return:
    """
    # 获取数据
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    print(text_list)
    # 文本特征转换
    transfer = CountVectorizer(stop_words=["一种", "今天"])
    new_data = transfer.fit_transform(text_list)

    # 查看特征名字
    names = transfer.get_feature_names()

    print("特征名字是：", names)
    print(new_data.toarray())  # 转换成one-hot矩阵
    print(new_data)


if __name__ == '__main__':
    # dict_demo()
    # english_count_demo()
    # chinese_count_demo()
    chinese_count_demo2()
