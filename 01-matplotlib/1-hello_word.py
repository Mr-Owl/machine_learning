import matplotlib.pyplot as plt
import random
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


def main():
    # 0.准备数据
    x = range(60)
    # uniform 生成一个随机实数
    y_shanghai = [random.uniform(15, 18) for i in x]
    y_beijing = [random.uniform(3, 10) for i in x]

    # 1.创建画布
    plt.figure(figsize=(20, 8), dpi=100)

    # 2.绘制图像
    plt.plot(x, y_shanghai, label="上海")
    plt.plot(x, y_beijing, color="r", linestyle="--", label="北京")

    # 2.1 添加x,y轴刻度
    # 构造x,y轴刻度标签
    x_ticks_label = ["11点{}分".format(i) for i in x]
    y_ticks = range(40)

    # 刻度显示,刻度数据不能直接改为字符串
    plt.xticks(x[::5], x_ticks_label[::5])
    plt.yticks(y_ticks[::5])

    # 2.2 添加网格显示
    plt.grid(True, linestyle="--", alpha=0.5)

    # 2.3 添加描述信息
    plt.xlabel("时间")
    plt.ylabel("温度")
    plt.title("中午11点--12点某城市温度变化图", fontsize=20)

    # 显示图例  loc显示位置 best自动设置位置
    plt.legend(loc="best")
    # 2.4 图像保存，在show之前保存
    # show之后会释放数据，就会变成一张空白图
    plt.savefig("./test.png")

    # 3.图像显示
    plt.show()


if __name__ == '__main__':
    main()
