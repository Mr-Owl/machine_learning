import numpy as np

if __name__ == '__main__':
    stock_change = np.random.normal(0, 1, (4, 5))
    # 形状修改
    # 1 ndarray.astype(type)
    # 返回修改了类型之后的数组
    print(stock_change)
    print(stock_change.astype(np.int64))
    # 2 ndarray.tostring([order])或者ndarray.tobytes([order])
    # 构造包含数组中原始数据字节的Python字节
    print("-----------------------------")
    print(stock_change)
    print(stock_change.tostring())
    #3 jupyter输出太大可能导致崩溃问题，需要修改配置文件