from matplotlib import pyplot as plt
import sys

sys.path.append('/home/caijz/temp/test_d2l/temp_d2l_pytorch')

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from downloadData import download
import os
from util.timer import Timer
import util.validate_macos as validate


# 房价预测 https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# 标准化的数据预处理步骤
def pre_process(all_features):
    # 1 处理数值型的缺失值
    # 1.0. 获取数字的列的index
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # print(numeric_features)

    # 1.1. 归一化
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # 1.2. 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 2 处理离散值（对象型）的缺失值，热独编码！！！
    # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指⽰符特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)

    # 3 整理数据
    # 3.0 获取训练集的行数
    n_train = train_data.shape[0]
    # print(n_train)
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    print("train feature size: " + str(train_features.shape))
    print("test feature size: " + str(test_features.shape))

    # 3.1 拿到最后一列作为标注
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    print("train label size: " + str(train_labels.shape))

    return train_features, test_features, train_labels


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值,将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


#
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 1. create iter
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # print("iter_size: " + train_iter.shape)
    # create optimizer 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_epoch = 0

    # 2. 在每个迭代周期(epoch)里,我们将完整遍历一次数据集(train_data),
    for epoch in range(num_epochs):
        num_epoch += 1
        num_batch = 0
        print(f'epoch No. {num_epoch} ')
        # 3. 不停地从中获取一个小批量(batch)的输入和相应的标签。
        for X, y in train_iter:
            num_batch += 1
            print(f'batch No. {num_batch} ')
            # print(f'X shape: {X.shape}')
            # print(f'y shape: {y.shape}')

            # 3.1 清空过往梯度
            optimizer.zero_grad()
            # 3.2 通过调用net(X)生成预测并计算损失l(前向传播)
            l = loss(net(X), y)
            print(l)
            # 3.3 通过进行反向传播来计算梯度
            l.backward()
            # 3.4 通过调用优化器来更新模型参数
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
        # 算每个迭代周期后的损失,并打印它来监控训练过程
        print(f'epoch {epoch + 1}, loss {l:f}')
    print(f'num_batch: {num_batch}')
    print(f'num_epoch: {num_epoch}')
    return train_ls, test_ls


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = validate.get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


#
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse:{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    # preds1 = net(test_features)
    preds = net(test_features).detach().numpy()


    # 保存模型
    model_path = os.path.join(cache_dir, 'house_price_model.pth')
    #1. 测试jit方式保存
    trace_model = torch.jit.trace(net, test_features)

    trace_model.save(model_path)

    #2. 测试save保存
    # torch.save(net, model_path)

    #3. 测试
    # model_read = torch.load(model_path)
    # print(model_read)


    # print(model.load_state_dict(weights))
    # print(model_path)
    # print(preds1)
    # print(preds1.shape)
    # print(preds)
    # print(preds.shape)
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    model_dir = os.path.join(cache_dir, 'submission.csv')
    print(model_dir)
    submission.to_csv(model_dir, index=False)


if __name__ == "__main__":
    timer = Timer()
    # timer.start()
    print("Kaggle实战：房价预测——数据下载")
    # 1. 下载训练集与测试集
    # 1.1 TODO: local dir for download 选择不同的电脑下载路径
    cache_dir = os.path.join('../../data', 'Kaggle_housePrice')
    # MBP path
    # cache_dir = os.path.join('/Users/quyuan/Desktop/Data', 'kaggle')

    # TODO: startdownload,，下载好之后就注释掉
    # train_data = pd.read_csv(download('kaggle_house_train', cache_dir))
    # test_data = pd.read_csv(download('kaggle_house_test', cache_dir))

    # 数据前处理
    # 1. 下载好之后，读取数据集
    train_data = pd.read_csv(os.path.join(cache_dir, 'kaggle_house_pred_train.csv'))
    test_data = pd.read_csv(os.path.join(cache_dir, 'kaggle_house_pred_test.csv'))
    # print(type(train_data))

    # 1.1 打印训练集测试集的形状
    # print(train_data.shape)
    # print(test_data.shape)
    # 1.2 打印前4个特征与最后的价格
    # print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
    # TODO：此处切掉了不带任何预测信息的ID，以及我们要预测的值那一列
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # print(all_features.shape)
    # print(all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

    # 1. data preprocess
    train_features, test_features, train_labels = pre_process(all_features)
    # 2. train
    # 2.1 define loss
    loss = nn.MSELoss()
    # get size of second dimension
    in_features = train_features.shape[1]
    # print(in_features)
    print("in_feature: " + str(in_features))
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
    #                           weight_decay, batch_size)
    # print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
    #       f'平均验证log rmse: {float(valid_l):f}')

    train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size)
    print(f'{timer.stop():.5f} sec')
