import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def main():
    print("ch3_7")
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    #1、定义模型
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    #2、定义参数
    net.apply(init_weights)
    #3. 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    #4. 定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == "__main__":
    main()
