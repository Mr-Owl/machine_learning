import numpy as np
import time


def timer(name):
    def test_time(func):
        def test_func(data):
            start = time.clock()
            result = func(data)
            end = time.clock()
            print(name + "耗时: %.3f" % (end - start))
            return result

        return test_func

    return test_time


@timer("list")
def sum_list(data):
    print(sum(data))


@timer("ndarry")
def sum_ndarray(data):
    print(data.sum())


if __name__ == '__main__':
    a = []
    for i in range(10000000):
        a.append(i)
    sum_list(a)
    b = np.array(a)
    sum_ndarray(b)
