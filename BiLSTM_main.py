from SST_dataset import SSTDataset
from GloVe_embedding import get_glove, glove_matrix
import numpy as np
import os
import argparse
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


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

    # token, embedding process
    args.add_argument('--max_sequence_length', type=int, default=40)
    args.add_argument('--embedding_dim', type=int, default=100)
    args.add_argument('--glove_dir', type=str, default='./Glove')

    config = args.parse_args()

    # Loading data
    train_data = SSTDataset(config.train_path, config.max_sequence_length)
    dev_data = SSTDataset(config.dev_path, config.max_sequence_length)
    test_data = SSTDataset(config.test_path, config.max_sequence_length)

    print('Total train dataset:   ', len(train_data))
    print('Total dev dataset:     ', len(dev_data))
    print('Total test dataset:    ', len(test_data))

    embedding_idx = get_glove(glove_path=config.glove_dir)
    embedding_matrix = glove_matrix(word_idx=train_data.word_index, embedding_idx=embedding_idx,
                                    embedding_dim=config.embedding_dim)

    # model build
    model = Sequential()
    model.add(Embedding(len(train_data.word_index) + 1, config.embedding_dim, weights=[embedding_matrix],
                        input_length=config.max_sequence_length, trainable=False))

    model.add(LSTM(128))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print()