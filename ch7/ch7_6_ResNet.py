import sys

sys.path.append('/home/caijz/temp/test_d2l/temp_d2l_pytorch')
import torch
from torch import nn
from d2l import torch as d2l
from util.timer import Timer
import matplotlib.pyplot as plt


def main():
    print("ch7_6 ResNet")


class Residual(nn.Module): #@save
    def __init__(self, input_channels, num_channels, use_1x1conv= False, strides = 1):
        super().__init__()
        # 输入输出不变
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)


if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'program cost {timer.stop():.5f} sec')
