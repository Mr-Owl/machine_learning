from sklearn.linear_model import LinearRegression

def main():
    # 1 获取数据
    x = [[80, 86],
         [82, 80],
         [85, 78],
         [90, 90],
         [86, 82],
         [82, 90],
         [78, 80],
         [92, 94]]
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
    # 2 数据基本处理(省略)
    # 3 特征工程-特征预处理(省略)
    # 4 模型训练
    estimator = LinearRegression()
    estimator.fit(x, y)
    # 打印对应的系数
    print("线性回归的系数:\n", estimator.coef_)
    # 打印的预测结果:
    print("输出预测结果:\n", estimator.predict([[100, 80]]))

    # 5 模型评估

if __name__ == '__main__':
    main()