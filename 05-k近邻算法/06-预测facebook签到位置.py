import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def main():
    """
    facebook签到位置预测
    数据来源：https://www.kaggle.com/c/facebook-v-predicting-check-ins
    :return:
    """
    # 1 获取数据集
    data = pd.read_csv("./data/FBlocation/train.csv")
    print(data.head())
    print(data.describe())
    print(data.shape)
    print("------------------")
    # 2 基本数据处理
    # 2.1 缩小数据范围
    partial_data = data.query("x > 2.0 & x < 2.5 & y > 2.0 & y < 2.5")
    print(partial_data.head())
    print(partial_data.shape)
    # 2.2 选择时间特征
    print(partial_data["time"].head())
    time = pd.to_datetime(partial_data["time"], unit="s")  # 将时间戳转换成datetime格式
    time = pd.DatetimeIndex(time)  # 将series格式的时间数据转换成DatatimeIndex格式，方便拿日期
    print(time.hour)
    partial_data["hour"] = time.hour
    partial_data["day"] = time.day
    partial_data["weekday"] = time.weekday
    print(partial_data.head())
    print("------------------")
    # 2.3 去掉签到较少的地方
    place_count = partial_data.groupby("place_id").count()
    place_count = place_count[place_count["row_id"] > 3]
    partial_data = partial_data[partial_data["place_id"].isin(place_count.index)]
    print(partial_data.shape)
    print("------------------")
    # 2.4 确定特征值和目标值
    x = partial_data[["x", "y", "accuracy", "hour", "day", "weekday"]]
    y = partial_data["place_id"]
    print(x)
    print(y)
    print("------------------")
    # 2.5 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    # 3 特征工程 -- 特征与处理（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    # 4 机器学习 -- knn+cv 交叉验证+网格搜索
    # 4.1 实例化一个训练模型
    estimator = KNeighborsClassifier()
    # 4.2 交叉验证，网格搜索，找到一个参数最好的模型
    param_grid = {"n_neighbors": [3, 5, 7, 9]}
    # n_jobs 用几个cpu核心跑 默认值为1。-1为全部
    estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, n_jobs=-1)
    # 4.3 模型训练
    estimator.fit(x_train, y_train)
    # 5 模型评估
    # 5.1 准确率输出
    score_ret = estimator.score(x_test, y_test)
    print("准确率为：", score_ret)
    # 5.2 预测结果
    y_pre = estimator.predict(x_test)
    print("预测值是：\n", y_pre)
    # 5.3 其他结果输出
    print("最好模型：\n", estimator.best_estimator_)
    print("最好模型参数：\n", estimator.best_params_)
    print("所有结果是：\n", estimator.cv_results_)


if __name__ == '__main__':
    main()
