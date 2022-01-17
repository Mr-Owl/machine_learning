import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    """
    回归决策树和线性回归对比
    :return:
    """
    # 特征数据是列数据，这里的range是行数据，所以要通过reshape转换成列式
    x = np.array(list(range(1, 11))).reshape(-1, 1)
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    # 模型训练
    m1 = DecisionTreeRegressor(max_depth=1)
    m2 = DecisionTreeRegressor(max_depth=3)
    m3 = LinearRegression()
    m1.fit(x, y)
    m2.fit(x, y)
    m3.fit(x, y)
    # 模型预测
    x_test = np.arange(0, 10, 0.01).reshape(-1, 1)
    y_1 = m1.predict(x_test)
    y_2 = m2.predict(x_test)
    y_3 = m3.predict(x_test)
    # 结果可视化
    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(x, y, label="data")
    plt.plot(x_test, y_1, label="max_depth=1")
    plt.plot(x_test, y_2, label="max_depth=3")
    plt.plot(x_test, y_3, label="linear regressoion")
    plt.xlabel("数据")
    plt.ylabel("预测值")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
