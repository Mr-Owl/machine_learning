import numpy as np

if __name__ == '__main__':
    # 矩阵  特殊的二维数组
    arr1 = np.array([[1, 2], [3, 4]])
    # 向量  一行或一列的数组
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([[1],
                     [2],
                     [3]])
    # 矩阵的加法：行列数相等可以加,对应位置的元素加对应位置的元素
    arr4 = np.random.randint(1, 10, (3, 2))
    arr5 = np.random.randint(1, 10, (3, 2))
    print(arr4 + arr5)
    print("-------------------------")
    # 矩阵与标量的乘法：每个元素与标量相乘
    # 矩阵乘法：axb  a矩阵的列数必须等于矩阵的行数
    # 性质：axb!=bxa,axbxc=ax(bxc)
    # 矩阵乘法遵循准则：(M行, N列)*(N行, L列) = (M行, L列)
    # c = a x b    c(ij) = a的第i行与b的第j列对应数字相乘的和

    # np.matmul 可以实现矩阵的乘法（禁止矩阵与标量的乘法，否则报错）
    arr6 = np.random.randint(40, 80, (5, 2))  # 代表5名同学的平时与期末成绩
    arr7 = np.array([[0.7],
                     [0.3]])  # 平时成绩与期末成绩的占比
    print(np.matmul(arr6, arr7))
    # print(np.matmul(arr6, 3))  # 不支持矩阵与标量的乘法会报错
    print("-------------------------")

    # np.dot 也可以实现矩阵的乘法（支撑矩阵与标量的乘法
    arr6 = np.random.randint(40, 80, (5, 2))
    print(arr6)
    print(np.dot(arr6, 2))

    # 矩阵的逆：a是一个方阵,如果有逆矩阵，那么：
    # a*a的逆矩阵 = 单列矩阵   那么他俩互为逆

    # 矩阵的转置：
    # A 为 m×n 阶矩阵（即 m 行 n 列），第 i 行 j 列的元素是 a(i,j)，即：
    # A=a(i,j)
    # 定义 A 的转置为这样一个 n×m 阶矩阵 B，
    # 满足 B=a(j,i)，即 b (i,j)=a (j,i)（B 的第 i 行第 j 列元素是 A 的第 j 行第 i 列元素），
    # 记 AT =B。
    # 简单的来说就是把该矩阵列当做成行第一列变为第一行，第n列变为第n行
