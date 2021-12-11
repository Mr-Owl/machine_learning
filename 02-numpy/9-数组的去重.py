import numpy as np

if __name__ == '__main__':
    a = np.array([[[1, 2, 3, 4],
                   [2, 3, 4, 5]],
                  [[3, 4, 5, 6],
                   [4, 5, 6, 7]]])
    print(a)
    # 降维去重
    print(np.unique(a))
