import torch
import os
import pandas as pd

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
    print(x)
    print(y)
    print(x>y)
    print(x<y)

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
    print(y)
    # [TODO]-2.1.5,2.1.6
    print('2.1.5 节省内存')



def book_ch2_2():
    print('2.2.1 数据预处理')
    #创建目录，存储CSV文件，如果目录不存在，则创建
    os.makedirs(os.path.join('/opt/data/dive_into_deep_learning', 'data'), exist_ok = True)
    data_file = os.path.join('/opt/data/dive_into_deep_learning', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        #按行写入
        f.write('NumRooms, Alley, Price\n') #列名
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')

    #利用pandas读取CSV文件的内容
    data = pd.read_csv(data_file)
    print(data)
    print('2.2.2 处理缺失值')
    print('插值法处理缺失值')
    inputs, outputs = data.iloc[:,0:2],data.iloc[:,2]
    #均值处理NumRooms
    inputs = inputs.fillna(inputs.mean())
    print(inputs)
    #将NaN视为一个类
    inputs = pd.get_dummies(inputs,dummy_na = True)
    print(inputs)

    print('2.2.3 转换为张量格式')
    tensor1,tensor2 = torch.tensor(inputs.values),torch.tensor(outputs.values)
    print(torch.ones((2,3)))
    print(tensor1)
    print(tensor2)

    print('homework:将预处理后的数据集转化为张量')
    tensor2 = torch.reshape(tensor2,(4,1))
    print(tensor2)
    data_result = torch.cat((tensor1,tensor2),dim = 1)
    print(data_result)
def book_ch2_3():
    #线性代数
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    print('2.3.1 标量')
    print(x+y,x-y,x*y,x/y)
    print('2.3.2 向量')
    x = torch.arange(4)
    print(x)
    print(x[3])
    print('长度为： ' + str(len(x)))
    print('形状为： ' + str(x.shape))
    print('2.3.3 矩阵')
    x = torch.arange(20).reshape(5,4)
    print(x)
    b = torch.tensor([[1,2,3],[2,0,5],[3,5,2]])

    print(b == b.T)

    print('2.3.5 张量')
    A = torch.arange(20,dtype=torch.float32).reshape(5,4)
    B = A.clone()
    print(A)
    print(A+B)
    print('Hadamard积： ' + str(A*B))

    x = torch.arange(4, dtype=torch.float32)
    print(x.sum())
    print(x.sum(axis=0))

    print('2.3.7 点积')
    y = torch.ones(4,dtype=torch.float32)
    x = torch.arange(4,dtype=torch.float32)
    print('点积： ' + str(torch.dot(x,y)))

    u = torch.tensor(([3.0,-4.0]))
    print('范数' + str(torch.norm(u)))



def main():
    print("Hello World!")


if __name__ == "__main__":
    main()
    book_ch2_1()
    #book_ch2_2()
    #book_ch2_3()
