# 暂退法
import torch
from d2l import torch as d2l
from torch import nn
def main():
    print("暂退法")
    # num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    # X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    # print(X)
    # print(dropout_Layer(X, 0.))
    # print(dropout_Layer(X, 0.5))
    # print(dropout_Layer(X, 1.))

    #设置丢弃概率
    dropout1, dropout2 = 0.2, 0.5
    #构建网络
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        #此处是第一个全连接层结束，再加入一个dropout层
                        nn.Dropout(dropout1),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        #此处是第二个全连接层结束，加入一个dropout层
                        nn.Dropout(dropout2),
                        nn.Linear(256,10)
                        )
    net.apply(init_weights);
    #训练
    num_epochs, lr, batch_size = 10, 0.5, 256
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def dropout_Layer(X, dropout_p):
    #暂退法的概率在0-1之间
    assert 0 <= dropout_p <=1
    # 在本情况中，所有元素都被丢弃
    if dropout_p == 1:
        result = torch.zeros_like(X)
        print(type(result))
        return result
    # 在本情况中，所有元素都被保留
    if dropout_p == 0:
        return X
    mask = (torch.rand(X.shape) > dropout_p).float()
    print("rand = %s" % (torch.rand(X.shape)) )
    print("mask = %s" % (mask))
    return mask * X / (1.0 - dropout_p)

#定义模型
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()


        def forward(self,X):
            H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))



#简洁实现




if __name__ == "__main__":
    main()
    # ch2_7_1()