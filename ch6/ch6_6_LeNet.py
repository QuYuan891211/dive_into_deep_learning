import sys

sys.path.append('/home/caijz/temp/test_d2l/temp_d2l_pytorch')
import os
import torch
from d2l import torch as d2l
from torch import nn
from torch import nn
from util.timer import Timer
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter


def main():
    print('ch6_6 LeNet')
    print('本节将介绍LeNet，它是最早发布的卷积神经网络之一，因其在计算机视觉任务中的高效性能而受到广泛关注。LeNet取得了与支持向量机（support vector '
          'machines）性能相媲美的成果，成为监督学习的主流方法, LeNet被广泛用于自动取款机（ATM）机中，帮助识别处理支票的数字。 时至今日，一些自动取款机仍在运行Yann LeCun和他的同事Leon '
          'Bottou在上世纪90年代写的代码呢！')
    # 创建编辑器，保存日志，指令保存路径log_dir
    # writer = SummaryWriter(log_dir="../logs")  # 指定保存位置
    net = nn.Sequential(
        # 卷积块1
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 卷积块2
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        # 为后续的全连接层展平为张量
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape: \t', X.shape)
    # writer.add_graph(model=net, input_to_model=X)
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.1, 3
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    # 保存模型
    cache_dir = os.path.join('../../data', 'LeNet')
    model_path = os.path.join(cache_dir, 'house_price_model.pth')
    #1. 测试jit方式保存
    trace_model = torch.jit.trace(net, test_iter)

    trace_model.save(model_path)
    plt.show()
    # writer.close()


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# @save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'program cost {timer.stop():.5f} sec')
