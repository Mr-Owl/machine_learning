import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

"""
特征工程之特征预处理:
通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程
为什么我们要进行归一化/标准化？
特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些算法无法学习到其它的特征

我们需要用到一些方法进行无量纲化，使不同规格的数据转换到同一规格

包含内容(数值型数据的无量纲化)
归一化
标准化

1，归一化
通过对原始数据进行变换把数据映射到(默认为[0,1])之间
公式：
先求出x1 = (x-min)/(max-min)
再求出x2 = x1*(xm-归一化区间的最小值)+mi
解释：作用于每一列，max为一列的最大值，min为一列的最小值,那么x2为最终结果，mx，mi分别为指定区间值默认mx为1,mi为0
归一化api:
①sklearn.preprocessing.MinMaxScaler (feature_range=(0,1)… )
feature_range指定归一化后的最大最小值
②MinMaxScalar.fit_transform(X)
X:numpy array格式的数据[n_samples,n_features]
返回值：转换后的形状相同的array
归一化总结：
注意最大值最小值是变化的，另外，最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。解决这个问题可以使用标准化


2,标准化
通过对原始数据进行变换把数据变换到均值为0,标准差为1范围内
公式：
x1=(x-mean)/σ
解释：作用于每一列，mean为平均值，σ为标准差

标准化和归一化在异常值处理的区别
对于归一化来说：如果出现异常点，影响了最大值和最小值，那么结果显然会发生改变
对于标准化来说：如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大，从而方差改变较小。

标准化api
1 sklearn.preprocessing.StandardScaler( ) 实例化对象
处理之后每列来说所有数据都聚集在均值0附近标准差差为
2 StandardScaler.fit_transform(X) 进行转换
X:numpy array格式的数据[n_samples,n_features]
返回值：转换后的形状相同的array

"""
def minmax_demo():
    """
    归一化演示
    :return:None
    """
    data = pd.read_csv("./data/dating.txt")
    print(data)
    # 1 实例化 确认归一化后的最大最小值
    transform = MinMaxScaler(feature_range=(3, 5))
    # 2 进行转换,调用fit_transform
    ret_data = transform.fit_transform(data[["milage", "Liters", "Consumtime"]])
    print("归一化后的数据:\n", ret_data)

def stand_demo():
    """
    标准化演示
    :return: None
    """
    data = pd.read_csv("./data/dating.txt")
    print(data)
    # 1 实例化
    transfer = StandardScaler()
    # 2 进行转换,调用fit_transform
    ret_data = transfer.fit_transform(data[["milage", "Liters", "Consumtime"]])
    print("标准化之后的数据为:\n", ret_data)
    print("每一列的方差为:\n", transfer.var_)
    print("每一列的平均值为:\n", transfer.mean_)


def main():
    minmax_demo()
    stand_demo()

if __name__ == '__main__':
    main()
