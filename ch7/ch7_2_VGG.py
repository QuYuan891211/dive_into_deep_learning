import sys

sys.path.append('/home/caijz/temp/test_d2l/temp_d2l_pytorch')
import torch
from torch import nn
from d2l import torch as d2l
from util.timer import Timer
import matplotlib.pyplot as plt

def main():
    print('ch7_2 VGG')

if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'program cost {timer.stop():.5f} sec')