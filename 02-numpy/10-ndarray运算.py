import numpy as np

if __name__ == '__main__':
    score = np.random.randint(40, 100, (10, 5))
    # 1 取后四行成绩用于逻辑判断
    test_score = score[6:, :]
    print(test_score)
    # 1.1把符合条件的设置成True，否则False
    test2_score = test_score > 60
    print(test2_score)
    print("---------------------")
    print(test_score)
    # 1.2修改符合条件的值为1
    test_score[test_score > 60] = 1
    print(test_score)
    print("---------------------")
    # 2通用函数判断
    # 2.1判断选中部分是否全满足条件,必须全部满足
    print(np.all(score[0:2, :] > 60))
    # 2.2判断选中部分是否有满足条件的，满足一个即可
    print(np.any(score[0:2, :] > 60))
    print("---------------------")
    # 3三元运算符
    # 3.1 np.where(数组逻辑判断, 值A，值B) 返回成立的改成A,否则改成B
    print(score[0:2, :])
    print(np.where(score[0:2, :] > 60, 1, 0))
    # 3.2 复合逻辑运算np.where条件部分可以用np.logical_and/or/not 开搭配使用,
    print(np.where(np.logical_and(score[0:2, :] > 60, score[0:2, :] < 90), 1, 0))
    print(np.where(np.logical_or(score[0:2, :] < 60, score[0:2, :] > 90), 1, 0))
    print(np.where(np.logical_not(score[0:2, :] > 60, score[0:2, :] < 90), 1, 0))
    print("---------------------")
    # 4 统计指标
    # 4.1 np.min/max/median(a. axis) 最小/最大/中位数(从小到大取中间的数，或中间两个数的平均值)
    # a代表数组，axis=0/1 代表行/列操作
    # axis=n，可以理解为多维数组有多个轴，根据数据在第n轴上的投影进行操作
    # 不传axis寻找所有中的一个，传axis值，按要求返回每行或每列中符合要求的
    temp = score[:4, :]
    print(temp)
    print(np.max(temp))
    print(np.min(temp))
    print(np.median(temp))
    print(np.max(temp, axis=0))  # 这里是按列
    print(np.min(temp, axis=1))  # 这里是按行
    print("----------------------------")
    # 4.2  np.argmax/min(a, axis)  返回最大/最小 下标
    print(temp)
    # 不传axis时， 返回的下表是把数组降为一维后的下标
    print(np.argmax(temp))
    print(np.argmin(temp, axis=0))
    # 4.3 np.mean/std/var(a, axis, dtype)  平均值/标准差/方差
    print(np.mean(temp, axis=1))
