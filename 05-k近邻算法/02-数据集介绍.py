import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# 设置显示中文字体
# 字体需要额外下载安装，另外配置后需要去家目录cache清除缓存
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
plt.rcParams["axes.unicode_minus"] = False


def main():
    # scikit-learn数据集API
    """
    sklearn.datasets
    加载获取流行数据集
    datasets.load_*()
    获取小规模数据集，数据包含在datasets里
    datasets.fetch_*(data_home=None)
    获取大规模数据集，需要从网络上下载，函数的第一个参数是data_home，表示数据集下载的目录,默认是 ~/scikit_learn_data/
    :return:
    """
    # 1 数据集获
    # 1.1 小数据集获取  本地的数据集
    # 这里加载的是自带的鸢尾花数据集 sklearn.datasets.load_iris()
    iris = load_iris()
    print(iris)
    print("---------------------------")
    # 1.2 大数据集获取  网络下载
    # sklearn.datasets.fetch_20newsgroups(data_home=None,subset=‘train’)
    # subset：'train'或者'test'，'all'，可选，选择要加载的数据集。
    # 训练集的“训练”，测试集的“测试”，两者的“全部”
    news = fetch_20newsgroups()
    print(news)
    # 2 sklearn数据集返回值介绍
    # load和fetch返回的数据类型datasets.base.Bunch(字典格式)
    # data：特征数据数组，是 [n_samples * n_features] 的二维 numpy.ndarray 数组
    # target：标签数组，是 n_samples 的一维 numpy.ndarray 数组
    # DESCR：数据描述
    # feature_names：特征名,新闻数据，手写数字、回归数据集没有
    # target_names：标签名
    # 鸢尾花为例
    print("鸢尾花数据集的返回值：\n", iris)
    # 返回值是一个继承自字典的Bench
    print("鸢尾花的特征值:\n", iris["data"])
    print("鸢尾花的目标值：\n", iris.target)
    print("鸢尾花特征的名字：\n", iris.feature_names)
    print("鸢尾花目标值的名字：\n", iris.target_names)
    print("鸢尾花的描述：\n", iris.DESCR)
    # 数据可视化
    """
    seaborn介绍
    Seaborn 是基于 Matplotlib 核心库进行了更高级的 API 封装，可以让你轻松地画出更漂亮的图形。而 Seaborn 的漂亮主要体现在配色更加舒服、以及图形元素的样式更加细腻。
    安装 pip3 install seaborn
    seaborn.lmplot() 是一个非常有用的方法，它会在绘制二维散点图时，自动完成回归拟合
    sns.lmplot() 里的 x, y 分别代表横纵坐标的列名,
    data= 是关联到数据集,
    hue=*代表按照 species即花的类别分类显示,
    fit_reg=是否进行线性拟合。 默认True
    api链接
    """
    iris_d = pd.DataFrame(data=iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
    iris_d["target"] = iris.target

    # print(iris_d)
    def iris_plot(data, col1, col2):
        sns.lmplot(x=col1, y=col2, data=data, hue="target", fit_reg=False)
        plt.title("鸢尾花数据展示")
        plt.show()

    iris_plot(iris_d, 'Sepal_Width', 'Petal_Length')
    iris_plot(iris_d, 'Sepal_Length', 'Petal_Width')

    """
    机器学习一般的数据集会划分为两个部分：

    训练数据：用于训练，构建模型
    测试数据：在模型检验时使用，用于评估模型是否有效
    划分比例：
    
    训练集：70% 80% 75%
    测试集：30% 20% 25%
    数据集划分api
    
    sklearn.model_selection.train_test_split(arrays, *options)
    参数：
    x 数据集的特征值
    y 数据集的标签值(目标值)
    test_size 测试集的大小，一般为float
    random_state 随机数种子,不同的种子会造成不同的随机采样结果。相同的种子采样结果相同。
    return:
    x_train, x_test, y_train, y_test
    训练集特征值,测试集特征值,训练集目标值,测试集目标值
    """
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值:\n", x_train)
    print("训练集的目标值:\n", y_train)
    print("测试集的特征值:\n", x_test)
    print("测试集的目标值:\n", y_test)

    print("训练集的目标值的形状:\n", x_train.shape)
    print("测试集的目标值的形状:\n", y_train.shape)

    x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
    print("测试集的目标值:\n", y_test)
    print("测试集的目标值:\n", y_test1)
    print("测试集的目标值:\n", y_test2)


if __name__ == '__main__':
    main()
