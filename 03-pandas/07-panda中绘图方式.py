import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # DataFrame.plot(kind='line')
    # kind : str，需要绘制图形的种类
    # ‘line’ : line plot (default) 折现
    # ‘bar’ : vertical bar plot  条形图
    # ‘barh’ : horizontal bar plot  横向条形图 竖轴为x,横轴为y
    # 关于“barh”的解释：
    # http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.barh.html
    # ‘hist’ : histogram  直方图
    # ‘pie’ : pie plot  饼状图
    # ‘scatter’ : scatter plot  散点图
    data = pd.DataFrame(np.random.randint(40, 100, (3, 2)),
                        index=["a", "b", "c"], columns=["chinese", "math"])
    data.plot(kind="barh")
    plt.show()