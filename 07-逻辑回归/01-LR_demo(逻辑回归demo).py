import pandas as pd

# 关闭ssl验证
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

"""
sklearn.linear_model.LogisticRegression(solver='liblinear', 
penalty=‘l2’, C = 1.0)

solver可选参数:{'liblinear', 'sag', 'saga','newton-cg', 'lbfgs'}，
默认: 'liblinear'；用于优化问题的算法。
对于小数据集来说，“liblinear”是个不错的选择，而“sag”和'saga'对于大型数据集会更快。
对于多类问题，只有'newton-cg'， 'sag'， 'saga'和'lbfgs'可以处理多项损失;
“liblinear”仅限于“one-versus-rest”分类。

penalty：正则化的种类
C：正则化力度

默认将类别数量少的当做正例
LogisticRegression方法相当于 SGDClassifier(loss="log", penalty=" "),
SGDClassifier实现了一个普通的随机梯度下降学习。
而使用LogisticRegression(实现了SAG)
"""
def main():
    # 原始数据的下载地址：https://archive.ics.uci.edu/ml/machine-learning-databases/
    names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
             'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=names)
    print(data)
if __name__ == '__main__':
    main()
