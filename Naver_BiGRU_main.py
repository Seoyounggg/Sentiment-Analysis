# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CO
ECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from NSM_dataset import NSMDataset, preprocess

from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. Py
    의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # Data path
    args.add_argument('--train_path', type=str, default='./Data/ratings_train.txt')
    args.add_argument('--dev_path', type=str, default='./Data/ratings_test.txt')
    args.add_argument('--test_path', type=str, default='./Data/test')

    # options
    args.add_argument('--max_sequence_length', type=int, default=30)
    args.add_argument('--embedding_dim', type=int, default=256)
    args.add_argument('--glove_dir', type=str, default='./Glove/glove.6B.300d.txt')
    args.add_argument('--lstm_size', type=int, default=5)

    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=60)
    args.add_argument('--lr', type=float, default=0.005)
    args.add_argument('--savemodel', type=bool, default=True)
    args.add_argument('--savename', type=str, default='BiLSTM_naver.h5')
    args.add_argument('--mode', type=str, default='train')

    config = args.parse_args()

    # Loading data
    train_data = NSMDataset(config.train_path, config.max_sequence_length)
    dev_data = NSMDataset(config.dev_path, config.max_sequence_length)

    print('Total train dataset:   ', len(train_data))
    print('Total dev dataset:     ', len(dev_data))

    inputs = layers.Input((config.max_sequence_length,))
    layer = layers.Embedding(251, config.embedding_dim, input_length=config.max_sequence_length)(inputs)
    layer = layers.Bidirectional(layers.CuDNNGRU(100, return_sequences=True))(layer)
    layer = layers.Bidirectional(layers.CuDNNGRU(100, return_sequences=False))(layer)

    layer1 = layers.Dense(2)(layer)
    outputs1 = layers.Activation('softmax')(layer1)

    layer2 = layers.Dense(1)(layer1)
    outputs2 = layers.Activation('sigmoid')(layer2)
    outputs2 = layers.Lambda(lambda layer: layer * 9 + 1)(outputs2)
    model = models.Model(inputs=inputs, outputs=[outputs1, outputs2])
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.001, amsgrad=True, clipvalue=1.0), loss=['categorical_crossentropy', 'mse'], metrics=['accuracy'])

    print(model.summary())

    # train
    if config.mode == 'train':

        train_one_batch = len(train_data) // config.batch
        dev_one_batch = len(dev_data) // config.batch
        best_acc = 0.0

        for epoch in range(config.epochs):

            avg_train_loss = 0.0
            avg_train_acc = 0.0
            train_data.shuffle()

            for i, (data, labels, sentiments) in enumerate(_batch_loader(train_data, config.batch)):
                loss, ce_loss, mse_loss, ce_acc, mse_acc = model.train_on_batch(data, [sentiments, labels])

                if i % 10 == 0:
                    print('Batch : ', i, '/', train_one_batch,
                          ', loss in minibatch: ', float(loss),
                          ', acc in minibatch: ', float(ce_acc),
                          'current best: ', best_acc)

                avg_train_loss += float(mse_loss)
                avg_train_acc += float(ce_acc)

                if i % 100 == 0:
                    avg_dev_acc = 0.0
                    dev_data.shuffle()

                    for j, (data_, labels_, sentiments_) in enumerate(_batch_loader(dev_data, config.batch)):
                        _, _, _, ce_acc, _ = model.test_on_batch(data_, [sentiments_, labels_])
                        avg_dev_acc += float(ce_acc)

                    cur_acc = avg_dev_acc / dev_one_batch

                    print('Epoch : ', epoch, 'Batch : ', i, '/', train_one_batch, 'Validation ACC : ', cur_acc)

                    if cur_acc >= best_acc and config.savemodel == True:
                        best_acc = cur_acc

                        print('###################  Best Acc Found  #############')
                        model.save('./modelsave/{}epoch'.format(epoch) + config.savename)
                        print('Save new model  {}epoch{}'.format(epoch, config.savename))

            print('\nEpoch: ', epoch, ' Train_loss: ', float(avg_train_loss / train_one_batch),
                  ' train_acc:', float(avg_train_acc / train_one_batch), '\n')

        print('best dev acc: ', best_acc)

    else:
        loadpath = './modelsave/' + '1epochBiLSTM.h5'
        model.load_weights(loadpath)

        test_data = NSMDataset(config.test_path, config.max_sequence_length)
        test_one_batch = len(test_data) // config.batch
        print('Total test dataset:    ', len(test_data))

        avg_test_acc = 0.0

        for k, (data_2, labels_2, sentiments_2) in enumerate(_batch_loader(test_data, config.batch)):
            _, _, _, ce_acc_test, _ = model.test_on_batch(data_2, [sentiments_2, labels_2])
            avg_test_acc += float(ce_acc_test)

        cur_acc = avg_test_acc / test_one_batch

        print('Test ACC : ', cur_acc)