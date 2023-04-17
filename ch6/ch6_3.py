import torch
from d2l import torch as d2l
from torch import nn
from util.timer import Timer
# from ch6_2 import corr2d
from util import gpu
def comp_conv2d(conv2d, X):
    # 这里的(1,1)表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度:批量大小和通道
    return Y.reshape(Y.shape[2:])
def main():
    print('ch6_3')
    # 请注意,这里每边都填充了1行或1列,因此总共添加了2行或2列
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1,device=gpu.try_gpu())
    X = torch.rand(size=(8, 8),device=gpu.try_gpu())
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1),device=gpu.try_gpu())
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2,device=gpu.try_gpu())
    print(comp_conv2d(conv2d, X).shape)

if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'program cost {timer.stop():.5f} sec')