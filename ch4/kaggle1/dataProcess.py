from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from downloadData import download

if __name__ == "__main__":
    print("Kaggle实战：房价预测——数据下载")
    # 1. 下载训练集与测试集，下载好之后就注释掉

    # train_data = pd.read_csv(download('kaggle_house_train'))
    # test_data = pd.read_csv(download('kaggle_house_test'))

    # 1. 下载好之后，读取数据集
    train_data = pd.read_csv("D:\data\kaggle\kaggle_house_pred_train.csv")
    test_data = pd.read_csv("D:\data\kaggle\kaggle_house_pred_test.csv")
    print(type(train_data))

    # 1.1 打印训练集测试集的形状
    print(train_data.shape)
    print(test_data.shape)
    # 1.2 打印前4个特征与最后的价格
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
