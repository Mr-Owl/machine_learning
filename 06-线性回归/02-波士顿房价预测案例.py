"""
# 1 获取数据
# 2 数据基本处理
# 2.1 分割数据
# 3 特征工程-标准化
# 4 机器学习-线性回归
# 5 模型评估
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error


def linear_model1():
    """
    线性回归：正规方程
    api:
    sklearn.linear_model.LinearRegression(fit_intercept=True)
    通过正规方程优化
    参数
    fit_intercept：是否计算偏置
    属性
    LinearRegression.coef_：回归系数
    LinearRegression.intercept_：偏置
    :return:
    """
    # 1 获取数据
    boston = load_boston()
    print(boston)
    # 2 数据基本处理
    # 2.1 分割数据
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4 机器学习-线性回归
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    print("模型的偏置是：\n", estimator.intercept_)
    print("模型的系数是：\n", estimator.coef_)
    # 5 模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print("预测值是：\n", y_pre)
    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("均方误差：\n", ret)


def linear_model2():
    """
    线性回归：梯度下降法
    sklearn.linear_model.SGDRegressor(loss="squared_loss",
    fit_intercept=True, learning_rate ='invscaling', eta0=0.01)
    SGDRegressor类实现了随机梯度下降学习，它支持不同的loss函数和正则化惩罚
    项来拟合线性回归模型。
    参数：
    loss:损失类型
    loss=”squared_loss”: 普通最小二乘法
    fit_intercept：是否计算偏置
    learning_rate : string, optional
    学习率填充
    'constant': eta = eta0
    'optimal': eta = 1.0 / (alpha * (t + t0)) [default]
    'invscaling': eta = eta0 / pow(t, power_t)
    power_t=0.25:存在父类当中
    对于一个常数值的学习率来说，可以使用learning_rate=’constant’ ，
    并使用eta0来指定学习率。
    属性：
    SGDRegressor.coef_：回归系数
    SGDRegressor.intercept_：偏置
    :return:
    """
    # 1 获取数据
    boston = load_boston()
    print(boston)
    # 2 数据基本处理
    # 2.1 分割数据
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4 机器学习-线性回归
    # estimator = SGDRegressor(max_iter=1000,
    #                          learning_rate="constant",
    #                          eta0=0.001)
    estimator = SGDRegressor(max_iter=1000)
    estimator.fit(x_train, y_train)
    print("模型的偏置是：\n", estimator.intercept_)
    print("模型的系数是：\n", estimator.coef_)
    # 5 模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print("预测值是：\n", y_pre)
    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("均方误差：\n", ret)


def linear_model3():
    """
    线性回归：岭回归
    sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True,solver="auto", normalize=False)
    具有l2正则化的线性回归
    alpha:正则化力度，也叫 λ
    λ取值：0~1 1~10

    solver:会根据数据自动选择优化方法
    sag:如果数据集、特征都比较大，选择该随机梯度下降优化

    normalize:数据是否进行标准化
    normalize=False:可以在fit之前调用preprocessing.StandardScaler标准化数据

    Ridge.coef_:回归权重
    Ridge.intercept_:回归偏置
    :return:
    """
    # 1 获取数据
    boston = load_boston()
    print(boston)
    # 2 数据基本处理
    # 2.1 分割数据
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
    # 3 特征工程-标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4 机器学习-线性回归
    # estimator = Ridge(alpha=1.0)
    estimator = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100))
    estimator.fit(x_train, y_train)
    print("模型的偏置是：\n", estimator.intercept_)
    print("模型的系数是：\n", estimator.coef_)
    # 5 模型评估
    # 5.1 预测值
    y_pre = estimator.predict(x_test)
    print("预测值是：\n", y_pre)
    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pre)
    print("均方误差：\n", ret)


if __name__ == '__main__':
    linear_model1()
    print("--------------------")
    linear_model2()
    print("--------------------")
    linear_model3()
