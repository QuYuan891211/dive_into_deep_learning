import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def reLU(x):
    y = torch.relu(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    plt.show()


def reLU_derivative(x):
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
    plt.show()


def sigmoid(x):
    y = torch.sigmoid(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(10, 2.5))
    plt.show()

def sigmoid_derivative(x):
    y = torch.sigmoid(x)
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
    plt.show()

def tanh(x):
    y = torch.tanh(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
    plt.show()

def tanh_derivative(x):
    y = torch.tanh(x)
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
def main():
    print("ch4_1")
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    tanh_derivative(x)


if __name__ == "__main__":
    main()
    # ch2_7_1()
