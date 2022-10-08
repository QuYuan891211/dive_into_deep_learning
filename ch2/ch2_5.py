import torch

def book_ch2_5():
    print("2.5.1")
    x = torch.arange(4.0)
    print(x)
    #专门设置一个属性存放梯度，而不是每次重新分配一个内存
    x.requires_grad_(True)
    print(x.grad)
    y = 2* torch.dot(x,x)
    print(y)
    y.backward()
    print(x.grad)

    x.grad.zero_()
    print(x.grad)

    y = x*x
    y.sum().backward()
    print(x.grad)

    print("2.5.3 分离计算")
    #只考虑x在y计算完之后产生的作用
    x.grad.zero_()
    y = x*x
    u = y.detach()
    z = u*x

    z.sum().backward()
    print(x.grad)

    x.grad.zero_()
    y.sum().backward()
    print(x.grad)




def main():
    print("ch2_5")


if __name__ == "__main__":
    main()
    #book_ch2_1()
    #book_ch2_2()
    book_ch2_5()
