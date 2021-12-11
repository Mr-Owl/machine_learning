import numpy as np

if __name__ == '__main__':
    # 1维索引，切片
    arr1 = np.arange(0, 10, 2)
    print(arr1[0])
    print(arr1[:2])
    # 2维索引，切片
    arr2 = np.random.normal(0, 1, (2, 5))
    print(arr2)
    # 第1个1维数组
    print(arr2[0, 2])
    print(arr2[0, :3])
    # 3维数组，索引
    arr3 = np.random.normal(0, 1, (2, 3, 4))
    print(arr3)
    print(arr3[0])
    print(arr3[1, 2, 0])
    print(arr3[1, 2, :2])
