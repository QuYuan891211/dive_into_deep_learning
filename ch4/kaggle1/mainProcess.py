from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from downloadData import download

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
    print(train_features.shape)
    print(test_features.shape)

    # 3.1 拿到最后一列作为标注
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    print(train_labels.shape)


if __name__ == "__main__":
    print("Kaggle实战：房价预测——数据下载")
    # 1. 下载训练集与测试集，下载好之后就注释掉

    # train_data = pd.read_csv(download('kaggle_house_train'))
    # test_data = pd.read_csv(download('kaggle_house_test'))

    # 数据前处理
    # 1. 下载好之后，读取数据集
    train_data = pd.read_csv("D:\data\kaggle\kaggle_house_pred_train.csv")
    test_data = pd.read_csv("D:\data\kaggle\kaggle_house_pred_test.csv")
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
    pre_process(all_features)
