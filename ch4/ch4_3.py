import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torch import nn


def main():
    print('ch4_3 多层感知机的简洁实现')
    batch_size, lr, num_epochs = 256, 0.1, 10

    #1. 读取数据
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 2. 初始化参数
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    # 3.定义模型 4. 激活函数
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    net.apply(init_weights);

    # 5. 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    #6. 定义梯度下降方案
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

if __name__ == "__main__":
    main()
    # ch2_7_1()
