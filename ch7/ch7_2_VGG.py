import sys

sys.path.append('/home/caijz/temp/test_d2l/temp_d2l_pytorch')
import torch
from torch import nn
from d2l import torch as d2l
from util.timer import Timer
import matplotlib.pyplot as plt


# 新建一个VGG块的方法，VGG其实相当于也是一种顺序块
def vgg_block(num_convs, in_channels, out_channels):
    # 一个VGG块包含多层
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    # * 用在实参上是拆包
    # 从nn.Sequential的定义来看，输入要么是orderdict,要么是一系列的模型，遇到list，必须用*号进行转化（将list迭代器进行拆包输出）
    # ，否则会报错 TypeError: list is not a Module subclass
    return nn.Sequential(*layers)


def vgg(conv_arch):
    # 多个块
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    # return这里相当于nn.Sequential（顺序块）嵌套，CH5中的内容
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 最后加入3个全连接层部分全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
    # 总结：最后加入3个连接层这里有改进空间
    # 最后总输出还是一个线性层


def main():
    print('ch7_2 VGG')
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)


if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'program cost {timer.stop():.5f} sec')
