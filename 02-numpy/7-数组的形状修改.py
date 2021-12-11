import numpy as np

if __name__ == '__main__':
    stock_change = np.random.normal(0, 1, (4, 5))
    print(stock_change)
    # 形状修改
    # 1 ndarray.reshape(shape, order)  shape 形状 oder 填充顺序，默认A 可看做行填充
    # 返回一个具有相同数据域，但shape不一样的视图
    # 行、列不进行互换
    print(stock_change.reshape([5, 4]))
    print(stock_change.shape)
    # 数组的形状被修改为: (2, 10), -1: 表示通过待计算 行 = 20/10
    print(stock_change.reshape([-1, 10]))
    print("---------------------------")
    # print(stock_change.reshape([3, -1]))  # 列=20/3，除不尽报错
    # 2 ndarray.resize(new_shape)
    # 修改数组本身的形状（需要保持元素个数前后相同）
    # 行、列不进行互换
    print(stock_change.shape)
    stock_change.resize([2, 10])  # 2行10列
    print(stock_change.shape)
    print("---------------------------")
    # ndarray.T
    # 数组的转置
    # 将数组的行、列进行互换
    print(stock_change)
    print(stock_change.T)

