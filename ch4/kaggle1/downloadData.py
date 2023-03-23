import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = (  # @save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (  # @save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
def download(name, cache_dir):  # @save
    # 下载一个DATA_HUB中的文件，返回本地文件名
    # assert name in DATA_HUB[name], f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    # 创建本地缓存路径(如果已存在就不再创建)
    os.makedirs(cache_dir, exist_ok=True)
    # 取url的最后一个下划线内容作为文件名
    fname = os.path.join(cache_dir, url.split('/')[-1])

    # 1. 如果缓存⽬录中已经存在此数据集⽂件，并且其sha-1与存储在DATA_HUB中的相匹配，我们将使⽤缓存的⽂件
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname) as f:
            while True:
                # 一次读取1MB（1024KB）
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存

    # 2. 缓存目录中不存在此数据集文件
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None): #@save
    #下载压缩
    fname = download(name)
    #获取文件名以外的根目录
    base_dir = os.path.dirname(fname)
    #分离文件名和扩展名
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar⽂件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  # @save
    # 下载DATA_HUB中的所有文件
    for name in DATA_HUB:
        download(name)


if __name__ == "__main__":
    print("Kaggle实战：房价预测——数据下载")
    # 1. 建立一个字典，将数据集名称映射到数据相关二元组上。二元组包括URL和sha-1密钥
    DATA_HUB = dict()
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
