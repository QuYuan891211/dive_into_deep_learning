import math
import torch
import numpy as np
from d2l import torch as d2l
from util.timer import Timer
from matplotlib import pyplot as plt

def ch3_1_2():
    print('3.1.2 矢量化加速')
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)
    timer = Timer()
    timer.start()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(c)
    print(f'矢量化加速前： {timer.stop(): .5f} sec')

    timer.start()
    d = a + b
    print(d)
    print(f'矢量化加速后： {timer.stop(): .5f} sec')

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)


def ch3_1_3():
    # 再次使用numpy进行可视化
    x = np.arange(-7, 7, 0.01)

    # 均值和标准差对
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
             ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    plt.show()


def main():
    print("ch3_1")

if __name__ == "__main__":
    main()
    #ch2_7_1()
    ch3_1_3()