import torch
from torch import nn
from util.timer import Timer


def main():
    print('ch5_6')
    # print(torch.device('cpu'))
    # print(torch.device('cuda'))
    # print(torch.device('cuda:1'))
    # print(torch.cuda.device_count())
    # print(try_gpu())
    # print(try_gpu(10))
    # print(try_all_gpus())
    # x = torch.tensor([1, 2, 3])
    # print(x.device)
    X = torch.ones(2, 3, device=try_gpu())
    # print(X)
    # Y = torch.rand(2, 3, device=try_gpu(1))
    # print(Y)
    # Z = Y.cuda(0)
    # print(f'X: {X}')
    # print(f'Z: {Z}')
    # print(X + Z)



    # error: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and
    # cpu! print(X + Y)

    net = nn.Sequential(nn.Linear(3, 1))
    net = net.to(device=try_gpu())
    print(net(X))
    # net[0].weight.data.device
    print(net[0].weight.data.device)
# 两个函数允许我们在不存在所需所有GPU的情况下运行代码
def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  # @save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'{timer.stop():.5f} sec')
