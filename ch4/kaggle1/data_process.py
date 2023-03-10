from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from download_data import download

if __name__ == "__main__":
    print("Kaggle实战：房价预测——数据下载")
    # 1. 建立一个字典，将数据集名称映射到数据相关二元组上。二元组包括URL和sha-1密钥

    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))
    print(train_data.shape)
    print(test_data.shape)
