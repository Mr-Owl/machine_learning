import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder


def main():
    """
    atto产品分类预测案例
    :return:
    """
    # 1 数据获取 奥托案例：https://www.kaggle.com/c/otto-group-product-classification-challenge/overview
    data = pd.read_csv("./data/otto/train.csv")
    # print(data.head())
    # print(data.shape)
    # print(data.describe())
    # 图形可视化查看数据分布
    # sns.countplot(data.target)
    # plt.show()
    # 由上图可以看出，该类比数据不均衡，需要后期处理
    # 2 数据基本处理
    # 数据已经脱敏，不再需要特殊处理
    # 2.1 截取部分数据（一般不要截取，这里电脑配置不够，所以截取）
    new1_data = data[:10000]
    # print(new1_data.shape)
    # sns.countplot(new_data.target)
    # plt.show()
    # 使用上面方式获取数据不可行，目标值类别减少的太多
    # 可使用随机欠采样获取数据
    # 首先需要确定特征值|标签值
    y = data["target"]
    x = data.drop(["id", "target"], axis=1)  # 删除列
    # print(x.head())
    # print(y.head())
    # 欠采样获取数据(为了减少电脑处理压力，这里用欠采样，一般情况我们使用过采样)
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(x, y)
    # print(x.shape, y.shape)
    # print(X_resampled.shape, y_resampled.shape)
    # sns.countplot(y_resampled)
    # plt.show()
    # 把标签值转换为数字
    le = LabelEncoder()
    y_resampled = le.fit_transform(y_resampled)
    # print(y_resampled)
    # 将数字转换为原来的标签值
    # print(le.inverse_transform(y_resampled))
    # 2.3 分割数据
    x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    # 3 模型训练
    # 3.1 基本模型训练
    rf = RandomForestClassifier(oob_score=True)  # 袋外估计，自助法
    rf.fit(x_train, y_train)
    y_pre = rf.predict(x_test)
    # print(y_pre)
    print("准确率:\n", rf.score(x_test, y_test))
    print("袋外估计:\n", rf.oob_score_)
    # sns.countplot(y_pre)
    # plt.show()
    # logloss 模型评估
    # log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    # 上面报错原因是因为log_loss 使用过程中，必须将输出用one-hot表示
    # 需要将这个多类别的输出结果通过OneHotEncoder修改为如下
    one_hot = OneHotEncoder(sparse=False)
    y_test1 = one_hot.fit_transform(y_test.reshape(-1, 1))
    y_pre1 = one_hot.fit_transform(y_pre.reshape(-1, 1))
    print("logloss:\n", log_loss(y_test1, y_pre1, eps=1e-15, normalize=True))
    # 因为这里预测结果直接输出的是1，0，而原本应该是可能性概率，所以上行输出logloss较大
    # 改变估计器中的预测值的输出模式，让输出结果为百分占比，降低logloss值
    y_pre_proba = rf.predict_proba(x_test)
    print("袋外估计:\n", rf.oob_score_)
    print("logloss:\n", log_loss(y_test1, y_pre_proba, eps=1e-15, normalize=True))
    # 3.2 模型调优
    # n_estimators, max_feature, maxdepth, min_samples_leaf
    # # 3.2.1 确定最优的n_estimators
    # # 确定n_estimators的取值范围
    # tuned_parameters = range(10, 200, 10)
    # # 创建添加accuracy的一个numpy
    # accuracy_t = np.zeros(len(tuned_parameters))
    # # 创建添加error的一个numpy
    # error_t = np.zeros(len(tuned_parameters))
    # # 调优过程实现
    # for j, one_parameter in enumerate(tuned_parameters):
    #     rf2 = RandomForestClassifier(n_estimators=one_parameter,
    #                                  max_depth=10,
    #                                  max_features=10,
    #                                  min_samples_leaf=10,
    #                                  oob_score=True,  # 是否使用袋外样本来估计泛化精度。默认False。
    #                                  random_state=0,
    #                                  n_jobs=1
    #                                  )
    #     rf2.fit(x_train, y_train)
    #     # 输出accuracy
    #     accuracy_t[j] = rf2.oob_score_
    #     # 输出logloss
    #     y_pre = rf2.predict_proba(x_test)
    #     error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    #     print(error_t)
    # # 优化结果过程可视化
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)
    # axes[0].plot(tuned_parameters, error_t)
    # axes[1].plot(tuned_parameters, accuracy_t)
    # axes[0].set_xlabel("n_estimators")
    # axes[0].set_ylabel("error_t")
    # axes[1].set_xlabel("n_estimators")
    # axes[1].set_ylabel("accuracy_t")
    # axes[0].grid(True)
    # axes[1].grid(True)
    # plt.show()
    # # 经过图像展示，最后确定n_estimators=175的时候表现不错
    # # 3.2.2 确定最优的max_features
    # tuned_parameters = range(5, 40, 5)
    # # 创建添加accuracy的一个numpy
    # accuracy_t = np.zeros(len(tuned_parameters))
    # # 创建添加error的一个numpy
    # error_t = np.zeros(len(tuned_parameters))
    # # 调优过程实现
    # for j, one_parameter in enumerate(tuned_parameters):
    #     rf2 = RandomForestClassifier(n_estimators=175,
    #                                  max_depth=10,
    #                                  max_features=one_parameter,
    #                                  min_samples_leaf=10,
    #                                  oob_score=True,  # 是否使用袋外样本来估计泛化精度。默认False。
    #                                  random_state=0,
    #                                  n_jobs=1
    #                                  )
    #     rf2.fit(x_train, y_train)
    #     # 输出accuracy
    #     accuracy_t[j] = rf2.oob_score_
    #     # 输出logloss
    #     y_pre = rf2.predict_proba(x_test)
    #     error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    # # 优化结果过程可视化
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)
    # axes[0].plot(tuned_parameters, error_t)
    # axes[1].plot(tuned_parameters, accuracy_t)
    # axes[0].set_xlabel("max_features")
    # axes[0].set_ylabel("error_t")
    # axes[1].set_xlabel("max_features")
    # axes[1].set_ylabel("accuracy_t")
    # axes[0].grid(True)
    # axes[1].grid(True)
    # plt.show()
    # # 经过图像展示，最后确定max_features=15的时候表现不错
    # # 3.2.3 确定最优的max_depth
    # tuned_parameters = range(10, 100, 10)
    # # 创建添加accuracy的一个numpy
    # accuracy_t = np.zeros(len(tuned_parameters))
    # # 创建添加error的一个numpy
    # error_t = np.zeros(len(tuned_parameters))
    # # 调优过程实现
    # for j, one_parameter in enumerate(tuned_parameters):
    #     rf2 = RandomForestClassifier(n_estimators=175,
    #                                  max_depth=one_parameter,
    #                                  max_features=15,
    #                                  min_samples_leaf=10,
    #                                  oob_score=True,  # 是否使用袋外样本来估计泛化精度。默认False。
    #                                  random_state=0,
    #                                  n_jobs=1
    #                                  )
    #     rf2.fit(x_train, y_train)
    #     # 输出accuracy
    #     accuracy_t[j] = rf2.oob_score_
    #     # 输出logloss
    #     y_pre = rf2.predict_proba(x_test)
    #     error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    # # 优化结果过程可视化
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)
    # axes[0].plot(tuned_parameters, error_t)
    # axes[1].plot(tuned_parameters, accuracy_t)
    # axes[0].set_xlabel("max_depth")
    # axes[0].set_ylabel("error_t")
    # axes[1].set_xlabel("max_depth")
    # axes[1].set_ylabel("accuracy_t")
    # axes[0].grid(True)
    # axes[1].grid(True)
    # plt.show()
    # # 经过图像展示，最后确定max_depth=30的时候表现不错
    # # 3.2.3 确定最优的min_samples_leaf
    # tuned_parameters = range(1, 10, 2)
    # # 创建添加accuracy的一个numpy
    # accuracy_t = np.zeros(len(tuned_parameters))
    # # 创建添加error的一个numpy
    # error_t = np.zeros(len(tuned_parameters))
    # # 调优过程实现
    # for j, one_parameter in enumerate(tuned_parameters):
    #     rf2 = RandomForestClassifier(n_estimators=175,
    #                                  max_depth=30,
    #                                  max_features=15,
    #                                  min_samples_leaf=one_parameter,
    #                                  oob_score=True,  # 是否使用袋外样本来估计泛化精度。默认False。
    #                                  random_state=0,
    #                                  n_jobs=1
    #                                  )
    #     rf2.fit(x_train, y_train)
    #     # 输出accuracy
    #     accuracy_t[j] = rf2.oob_score_
    #     # 输出logloss
    #     y_pre = rf2.predict_proba(x_test)
    #     error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)
    # # 优化结果过程可视化
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)
    # axes[0].plot(tuned_parameters, error_t)
    # axes[1].plot(tuned_parameters, accuracy_t)
    # axes[0].set_xlabel("min_samples_leaf")
    # axes[0].set_ylabel("error_t")
    # axes[1].set_xlabel("min_samples_leaf")
    # axes[1].set_ylabel("accuracy_t")
    # axes[0].grid(True)
    # axes[1].grid(True)
    # plt.show()
    # # 经过图像展示，最后确定min_samples_leaf=1的时候表现不错
    # 3.3 确定最优模型
    # n_estimatros = 175,max_depth = 30,max_features = 15,min_samples_leaf=1
    rf3 = RandomForestClassifier(n_estimators=175,
                                 max_depth=30,
                                 max_features=15,
                                 min_samples_leaf=1,
                                 oob_score=True,
                                 random_state=40,
                                 n_jobs=1
                                 )
    rf3.fit(x_train, y_train)
    print("优化模型准确率：\n", rf3.score(x_test, y_test))
    print("优化模型袋外估计：\n", rf3.oob_score_)
    print("优化模型logloss：\n", log_loss(y_test, rf3.predict_proba(x_test), eps=1e-15, normalize=True))
    # 4 生成提交数据
    test_data = pd.read_csv("./data/otto/test.csv")
    # print(test_data.head())
    test_data_drop_id = test_data.drop(["id"], axis=1)
    # print(test_data_drop_id.head())
    y_pre_test = rf3.predict_proba(test_data_drop_id)
    result_data = pd.DataFrame(y_pre_test, columns=["Class_" + str(i) for i in range(1, 10)])
    print(result_data.head())
    result_data.insert(loc=0, column="id", value=test_data.id)
    print(result_data.head())
    result_data.to_csv("./data/otto/submission.csv", index=False)


if __name__ == '__main__':
    main()
