import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
from util.timer import Timer



class MLP(nn.Module):
    # 用模型参数声明层。这里,我们声明两个全连接的层(1 hidden layer, 1 output layer)
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样,在类实例化时也可以指定其他函数参数,例如模型参数params(稍后将介绍)
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

        # 定义模型的前向传播,即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意,这里我们使用ReLU的函数版本,其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):

        # 这里,module是Module子类的一个实例。我们把它保存在'Module'类的成员
        # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module
            print(self._modules)

    def forward(self, X):

        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
def main():
    # 通过实例化nn.Sequential来构建我们的模型,层的执行顺序是作为参数传递的。简而言
    # 之,nn.Sequential定义了一种特殊的Module,即在PyTorch中表示一个块的类,它维护了一个由Module组成
    # 的有序列表。
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    # test data
    X = torch.rand(2, 20)

    print('自定义块: ')
    net2 = MLP()
    output2 = net2(X)
    # print(net(X))
    print(output2)
    print('顺序块: ')
    net3 = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    output3 = net3(X)
    print(output3)


if __name__ == "__main__":
    timer = Timer()
    main()
    print(f'{timer.stop():.5f} sec')
    # ch2_7_1()
