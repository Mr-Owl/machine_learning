import matplotlib.pyplot as plt
from pylab import mpl

# 设置显示中文字体
# 字体需要额外下载安装，另外配置后需要去家目录cache清除缓存
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


def main():
    # 准备数据
    x = [1, 2, 3, 4, 5]
    # 各部分扇形名称
    labels = ['a', 'b', 'c', 'd', 'e']
    # 各部分扇形距离中心的距离 百分比 想把哪个扇形分离出来就把对应位置改成大于0的数
    explode = [0, 0, 0, 0, 0.1]
    # 创建画布
    plt.figure(figsize=(20, 8), dpi=100)
    # labels 扇形名称，  autpct 百分比位数显示    explode 扇形对于中心的距离
    plt.pie(x, labels=labels, autopct="%.2f%%", explode=explode)
    # 设置xy轴刻度等长 让坐标系变成正方形，使圆饼图变圆
    plt.axis('equal')
    # 展示
    plt.show()


if __name__ == '__main__':
    main()
