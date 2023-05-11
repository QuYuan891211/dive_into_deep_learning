import torch
import d2l.torch as d2l
from torch import nn
from util.timer import Timer
from util import gpu

# """计算二维互相关运算"""
def corr2d(X, K):    #@save
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), device=gpu.try_gpu())
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
def main():
    print('ch6_2')
    # timer_cpu = Timer()
    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    # K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    # print(X)
    # print(K)
    #
    # print(corr2d(X, K))
    # print(f'CPU time is {timer_cpu.stop():.5f} sec')
    # timer_GPU = Timer()
    # X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], device=gpu.try_gpu())
    # K = torch.tensor([[0.0, 1.0], [2.0, 3.0]],device=gpu.try_gpu())
    # print(X)
    # print(K)
    #
    # print(corr2d(X, K))
    # print(f'GPU time is {timer_GPU.stop():.5f} sec')

    # segment detect
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    # print(X)
    K = torch.tensor([[1.0, -1.0]])
    # print(K)
# def a conv-layer
    Y = corr2d(X, K)
    # print(Y)
    # 构造一个二维卷积层,它具有1个输出通道和形状为(1,2)的卷积核
    # 这个二维卷积层使用四维输入和输出格式(批量大小、通道、高度、宽度),
    # segment detect
    X = torch.ones((6, 8), device=gpu.try_gpu())
    X[:, 2:6] = 0
    # print(X)
    K = torch.tensor([[1.0, -1.0]], device=gpu.try_gpu())
    # print(K)
# def a conv-layer
    Y = corr2d(X, K)
    # print(Y)
    # 构造一个二维卷积层,它具有1个输出通道和形状为(1,2)的卷积核
    # 这个二维卷积层使用四维输入和输出格式(批量大小、通道、高度、宽度),
    timer_gpu = Timer()
    net = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
    net = net.to(device=gpu.try_gpu())
    X = X.reshape((1,1,6,8))
    # Y size is ((X.shape[0] - K.shape[0] + 1), (X.shape[1] - K.shape[1] + 1)
    Y = Y.reshape((1,1,6,7))
    lr = 3e-2  # 学习率
    print(X)
    print(Y)

    for i in range(10):
        Y_hat = net(X)
        loss = (Y - Y_hat)**2
        net.zero_grad()
        loss.sum().backward()
        # 迭代卷积核
        net.weight.data[:] -= lr * net.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {loss.sum():.3f}')

    kernel = net.weight.data.reshape((1, 2))
    print(f'device is {kernel.device}')
    print(f'kernel is {kernel}')
    print(f'GPU time is {timer_gpu.stop():.5f} sec')


class conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'program cost {timer.stop():.5f} sec')