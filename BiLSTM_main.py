from SST_dataset import SSTDataset
from GloVe_embedding import get_glove, glove_matrix
import numpy as np
import os
import argparse
from keras import models, layers, optimizers


def _batch_loader(iterable, n=1):
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # Data path
    args.add_argument('--train_path', type=str, default='./Data/train')
    args.add_argument('--dev_path', type=str, default='./Data/dev')
    args.add_argument('--test_path', type=str, default='./Data/test')

    config = args.parse_args()

    print('Loading data')

    train_data = SSTDataset(config.train_path)
    dev_data = SSTDataset(config.dev_path)
    test_data = SSTDataset(config.test_path)

    print('Total number of train dataset:   ', len(train_data))
    print('Total number of dev dataset:     ', len(dev_data))
    print('Total number of test dataset:    ', len(test_data))

    print()