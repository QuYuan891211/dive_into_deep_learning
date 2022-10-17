import random
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def synthetic_data(w, b, num_examples):  # @save
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def ch3_2_1():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # print('features: ', features)
    # print('labels: ', labels)
    d2l.set_figsize()
    d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
    # plt.show()
    # len表示第0维的长度
    # print(len(features))
    return features, labels


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


def ch3_2_2():
    batch_size = 10
    features, labels = ch3_2_1()
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break


def ch3_2_3():
    print('3.2.3 初始化参数')
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    print('w：', w)
    print('b: ', b)

def linreg(X, w, b): #@save
    print('定义线性回归模型')
    return torch.matmul(X,w) + b

def squared_loss(y_hat, y):
    print('SE')
    return (y_hat - y.reshape(y_hat.shape))**2/2

def sgd(params, lr, batch_size): #@save
    with torch.no_grad():
        for param in params:
            param -=lr * param.grad / batch_size
            param.grad.zero_()

def main():
    print("ch3_1")


if __name__ == "__main__":
    main()
    # ch2_7_1()
    ch3_2_2()
