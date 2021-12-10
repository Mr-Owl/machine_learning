import matplotlib.pyplot as plt
import random
from pylab import mpl

# 设置显示中文字体
# 字体需要额外下载安装，另外配置后需要去家目录cache清除缓存
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


def main():
    # 0.准备数据
    x = range(60)
    # uniform 生成一个随机实数
    y_shanghai = [random.uniform(15, 18) for i in x]
    y_beijing = [random.uniform(3, 10) for i in x]

    # 1.创建画布 创建norws行ncols列个坐标系  fig 图对象  axes 对应数量的坐标系
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), dpi=100)
    # 2.绘制图像
    axes[0].plot(x, y_shanghai, label="上海")
    axes[1].plot(x, y_beijing, color="r", linestyle="--", label="北京")

    # 2.1 添加x,y轴刻度
    # 构造x,y轴刻度标签
    x_ticks_label = ["11点{}分".format(i) for i in x]
    y_ticks = range(40)
    #
    # 1坐标系添加x刻度
    axes[0].set_xticks(x[::5])
    # 1坐标系修改x带字符串的刻度
    axes[0].set_xticklabels(x_ticks_label[::5])
    # 1坐标系添加y刻度
    axes[0].set_yticks(y_ticks[::5])
    # 2坐标系添加刻度
    axes[1].set_xticks(x[::5])
    axes[1].set_xticklabels(x_ticks_label[::5])
    axes[1].set_yticks(y_ticks[::5])

    # 2.2 添加网格显示
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    #
    # 2.3 添加多坐标系描述信息
    axes[0].set_xlabel("时间")
    axes[0].set_ylabel("温度")
    axes[0].set_title("中午11点--12点某城市温度变化图", fontsize=20)
    axes[1].set_xlabel("时间")
    axes[1].set_ylabel("温度")
    axes[1].set_title("中午11点--12点某城市温度变化图", fontsize=20)

    #
    # 显示图例  loc显示位置 best自动设置位置
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    # # 2.4 图像保存，在show之前保存
    # show之后会释放数据，就会变成一张空白图
    # plt.savefig("./test.png")

    # 3.图像显示
    plt.show()


if __name__ == '__main__':
    main()
