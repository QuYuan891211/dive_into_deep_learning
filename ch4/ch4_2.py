import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torch import nn


def main():
    print("ch4_2")

    # 1. 读取数据，赋值到训练迭代器和测试迭代器上
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    num_epochs, lr = 10, 0.1

    # 2. 参数初始化
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]
    updater = torch.optim.SGD(params, lr=lr)

    # 3. 定义激活函数
    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)

    # 4. 定义模型
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)
        return H @ W2 + b2

    # 5. 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 6. 训练
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    d2l.predict_ch3(net, test_iter)

if __name__ == "__main__":
    main()
    # ch2_7_1()
