import numpy
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


def synthetic_data(true_w, true_b):
    print("3.3.1 生成数据集")

    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    return features, labels


def load_array(data_arrays, batch_size, is_train=True):  # @save
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def read_data(features, labels):
    batch_size = 10

    print("3.3.2 读取数据集")
    data_iter = load_array((features, labels), batch_size)
    # print("人工模拟的数据集（X，Y）：", next(iter(data_iter)))
    return data_iter


def init_model():
    print("3.3.3 定义一个线性回归模型（全连接）")
    net = nn.Sequential(nn.Linear(2, 1))
    print("初始化w, b")
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    return net


def init_loss():
    print("3.3.5 定义损失函数")
    loss = nn.MSELoss()
    return loss


def init_optim(net):
    print("3.3.6 定义优化算法")
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    return trainer


def train():
    print("3.3.7 训练")
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    num_epochs = 3
    features, labels = synthetic_data(true_w, true_b)
    data_iter = read_data(features, labels)
    net = init_model()
    loss = init_loss()
    trainer = init_optim(net)

    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)


def main():
    print("ch3_1")


if __name__ == "__main__":
    main()
    # ch2_7_1()
    train()
