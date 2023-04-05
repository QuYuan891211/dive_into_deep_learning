import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
from util.timer import Timer


def main():
    print('ch5_2')
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    print(net(X))
    print(net[2].state_dict())
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])
    net.state_dict()['2.bias'].data
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))

    print(rgnet(X))
    print(rgnet)
    print(rgnet[0][1][0].bias.data)

    net.apply(init_normal)
    print(net[0].weight.data[0], net[0].bias.data[0])

    net.apply(init_constant)
    print(net[0].weight.data[0], net[0].bias.data[0])
# block factory
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
    # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'{timer.stop():.5f} sec')
    # ch2_7_1()
