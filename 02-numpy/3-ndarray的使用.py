import numpy as np

if __name__ == '__main__':
    score = np.array(
        [[80, 89, 86, 67, 79],
         [78, 97, 89, 67, 81],
         [90, 94, 78, 67, 74],
         [91, 91, 90, 67, 69],
         [76, 87, 75, 67, 86],
         [70, 79, 84, 67, 84],
         [94, 92, 93, 67, 64],
         [86, 85, 83, 67, 80]])
    # 属性
    # 获取数组维度
    print(score.shape)
    # 数组维数
    print(score.ndim)
    # 数组元素数量
    print(score.size)
    # 一个元素的大小（长度，多少个字节）
    print(score.itemsize)
    # 数组元素的类型 (数据类型)
    print(score.dtype)
    # 1维数组 (3,)  3列
    a = np.array([1, 2, 3])
    print(a.shape)
    # 2维数组 (2,3)  2行3列
    b = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(b.shape)
    # 3维数组 (2,2,3) 2个2维数组，一个2维数组里有2个一位数组，一个1维数组有3个元素
    c = np.array([[[1, 2, 3],
                   [4, 5, 6]],
                  [[7, 8, 9],
                   [1, 2, 3]]])
    print(c.shape)
    # 数组的类型
    # 创建数组可以通过dtype，指定数组的类型  float32 单精度浮点数
    d = np.array([1, 2, 3], dtype=np.float32)
    print(d)
    # np.string_ 字符串 可以简写成"S"
    e = np.array(["i", 'like', 'python'], dtype=np.string_)
    print(e)  # 交互式直接a 会输出 dtype ='|S6' 6代表最长的字符串长度6
