import numpy as np

if __name__ == '__main__':
    # 生成0和1的数组
    # 生成都是1的数组 np.ones(shape, dtype)/ no.ones_like(a, dtype)
    ones = np.ones([4, 8])  # 指定维度
    print(ones)
    # 生成都是0的数组 np.zeros(shape, dtype)/ no.zeros_like(a, dtype)
    zeros = np.zeros_like(ones)  # 根据传入的数组生成相同维度样式0的数组
    print(zeros)
    # 从现有数组中生成一样的数组
    a = np.array([[1, 2, 3], [4, 5, 6]])
    # np.array(a)  深拷贝  a变 a1不会变
    a1 = np.array(a)
    # np.asarray(a) 浅拷贝  a变  a2会变
    a2 = np.asarray(a)
    # 修改a 二维数组中的第1个一维数组的第一个元素
    a[0, 0] = 1000
    print(a1)  # 没变
    print(a2)   # 变了
    # 生成等差数组
    # 等差数列---指定数量 start,stop,num数量默认50,endpoint是否包含stop,默认True
    print(np.linspace(0,10,5))
    print()
    # 等差数列---指定补偿 start,stop取不到,step,dtype
    print(np.arange(0,10,2))
    print()
    # 等比数列   start,stop,num默认50  10^x次方序列等比数列
    print(np.logspace(0,2,3))
