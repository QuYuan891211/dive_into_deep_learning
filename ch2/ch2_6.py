import torch
from torch.distributions import multinomial
from d2l import torch as d2l

def ch_2_6_1():
    #定义一个概率向量（模拟执筛子）
    fair_probs = torch.ones([6])/6
    print(multinomial.Multinomial(1, fair_probs).sample())
    count = multinomial.Multinomial(1000, fair_probs).sample()
    print(count/1000)
    counts = multinomial.Multinomial(10, fair_probs).sample((500,))
    cum_counts = counts.cumsum(dim=0)
    estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

    d2l.set_figsize((6, 4.5))
    for i in range(6):
        d2l.plt.plot(estimates[:, i].numpy(),
                     label=("P(die=" + str(i + 1) + ")"))
    d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
    d2l.plt.gca().set_xlabel('Groups of experiments')
    d2l.plt.gca().set_ylabel('Estimated probability')
    d2l.plt.legend();
    
def main():
    print("ch2_6")


if __name__ == "__main__":
    main()
    ch_2_6_1()