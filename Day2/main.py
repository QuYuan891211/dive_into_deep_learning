import torch


def book_ch2_1():
    x = torch.arange(12)
    print('2.1.1 获取一个张量（vector）')
    print(x)
    print('获取张量的大小')
    print(x.numel())
    print('改变张量的形状')
    print(x.reshape(3, 4))
    print('全为1的张量')
    print(torch.ones(2, 3, 4))
    print('随机数张量，0-1之间的高斯分布')
    print(torch.randn(3, 4))

    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2,2,2,2])
    print('2.1.2 矩阵四则运算')
    print(x+y, x-y,x*y,x/y,x**y)
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
    print('矩阵连接, 按照行进行拼接和按照列进行拼接')
    print(torch.cat((x,y),dim=0))
    print(torch.cat((x,y),dim=1))

    print('逻辑张量')
    print(x==y)

    print('张量元素求和')
    print(x.sum())

    a = torch.arange(3).reshape((3,1))
    b = torch.arange(2).reshape((1,2))

    print('2.1.3 广播机制')
    print(a,b)
    print(a + b)

    print('2.1.4 索引和碎片')
    print(x[-1],x[1:3])
    print(x)
    x[1,2] = 9
    print(x)
    x[0:2,:] = 12
    print(x)

    print('2.1.5 节省内存')
    
def main():
    print("Hello World!")


if __name__ == "__main__":
    main()
    book_ch2_1()
