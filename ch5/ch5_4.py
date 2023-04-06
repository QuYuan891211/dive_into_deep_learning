import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
from util.timer import Timer

class createLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def  forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
def main():

    print('ch5_4')
    layer = createLayer()
    print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

    linear = MyLinear(5, 3)
    print(linear.weight)
    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))
if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'{timer.stop():.5f} sec')