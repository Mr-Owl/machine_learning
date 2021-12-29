from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

"""
交叉验证：
将拿到的训练数据，分为训练和验证集。
举例：
将训练集数据分成4份，其中一份作为验证集。然后经过4次(组)的测试，每次都更换不同的验证集。
即得到4组模型的结果，取平均值作为最终结果。又称4折交叉验证。

训练集：训练集+验证集
测试集：测试集

交叉验证目的：为了让被评估的模型更加准确可信
这个只是让被评估的模型更加准确可信，那么怎么选择或者调优参数呢？

网格搜索:
通常情况下，有很多参数是需要手动指定的（如k-近邻算法中的K值），
这种叫超参数。但是手动过程繁杂，所以需要对模型预设几种超参数组合。
每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。

交叉验证，网格搜索（模型选择与调优）API：

sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)
    对估计器的指定参数值进行详尽搜索
    estimator：估计器对象
    param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
    cv：指定几折交叉验证
    
    fit：输入训练数据
    score：准确率
    结果分析：
        best_score_:在交叉验证中验证的最好结果
        best_estimator_：最好的参数模型
        best_params_: 最好模型参数
        cv_results_:每次交叉验证后的验证集准确率结果和训练集准确率结果
"""


def main():
    # 1 获取数据
    iris = load_iris()
    # 2 数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    # 3 特征工程 - 特征预处理(无量纲化)
    # 3.1 实例化对象转换器对象
    transfer = StandardScaler()
    # 3.2 转换
    x_train = transfer.fit_transform(x_train)
    # x_test = transfer.fit_transform(x_test)
    x_test = transfer.transform(x_test)
    # 4 机器学习-knn
    # 4.1 实例化估计器
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 4.2 模型调优 -- 交叉验证,网格搜索
    param_grid = {"n_neighbors": [1, 3, 5, 7]}
    estimator = GridSearchCV(estimator, param_grid=param_grid, cv=5)
    # 4.2 模型训练
    estimator.fit(x_train, y_train)
    # 5 模型评估
    # 5.1预测结果输出
    y_pre = estimator.predict(x_test)
    print("预测值是:\n", y_pre)
    print("预测值和真实值的对比时是:\n", y_pre == y_test)
    score = estimator.score(x_test, y_test)
    print("准确率为:", score)
    # 5.3 查看交叉验证,网格搜索的一些属性
    print("在交叉验证中,得到的最好结果是:\n", estimator.best_score_)
    print("在交叉验证中,得到的最好参数模型是:\n", estimator.best_estimator_)
    print("在交叉验证中,得到的最好参数是:\n", estimator.best_params_)
    print("在交叉验证中,得到的模型结果是:\n", estimator.cv_results_)


if __name__ == '__main__':
    main()
