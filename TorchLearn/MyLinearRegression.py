import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

lr = 0.05  # 学习率

# 创建训练数据
x = torch.rand(20, 1) * 10  # x data (tensor), shape=(20, 1)
y = 2*x + (5 + torch.randn(20, 1))  # y data (tensor), shape=(20, 1)

# 构建线性回归参数
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):

    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算 MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数 subtract
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 清零张量的梯度
    w.grad.zero_()
    b.grad.zero_()

    # clf()  # 清图。
    # cla()  # 清坐标轴。
    # close()  # 关窗口

    # 绘图
    if iteration % 20 == 0:

        #构建真实值得散点图
        plt.scatter(x.data.numpy(), y.data.numpy())
        #x,y 后面一个参数表示颜色-用什么样式的线连接两点-用什么表示点，最后一个表示线的宽度
        # plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.plot(x.data.numpy(), y_pred.data.numpy(),'m-', lw=2)
        #xy:文字开始的位置,string:表示文字，大小颜色等
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        #xlim与ylim（设置坐标轴数据范围
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        #图像抬头
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()),)
        #显示秒数
        plt.pause(0.5)
        plt.savefig(f'{iteration}.jpg')
        plt.cla()   #清图

        if loss.data.numpy() < 1:
            break
