from NSM_dataset import NSMDataset, preprocess
import argparse
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, GlobalMaxPooling1D, Dropout
from keras import optimizers, layers, models


def _batch_loader(iterable, n=1):
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # Data path
    args.add_argument('--train_path', type=str, default='../Data/ratings_train.txt')
    args.add_argument('--dev_path', type=str, default='../Data/ratings_test.txt')
    args.add_argument('--test_path', type=str, default='../Data/ratings_test.txt')

    # options
    args.add_argument('--max_sequence_length', type=int, default=30)
    args.add_argument('--embedding_dim', type=int, default=256)

    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=60)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--savemodel', type=bool, default=True)
    args.add_argument('--savename', type=str, default='korean_CNN.h5')
    args.add_argument('--mode', type=str, default='train')

    config = args.parse_args()


    # Loading data
    train_data = NSMDataset(config.train_path, config.max_sequence_length)
    dev_data = NSMDataset(config.dev_path, config.max_sequence_length)

    print('Total train dataset:   ', len(train_data))
    print('Total dev dataset:     ', len(dev_data))


    # model build

    inputs = layers.Input((config.max_sequence_length,))
    layer = layers.Embedding(252, config.embedding_dim, input_length=config.max_sequence_length)(inputs)
    layer = Conv1D(256, 3, padding='valid', activation='relu', strides=1)(layer)
    layer = GlobalMaxPooling1D()(layer)
    layer = Dense(128, activation='relu')(layer)

    outputs1 = layers.Dense(2, activation='softmax')(layer)
    
    outputs2 = layers.Dense(1, activation='sigmoid')(layer)
    outputs2 = layers.Lambda(lambda layer: layer * 9 + 1)(outputs2)
    model = models.Model(inputs=inputs, outputs=[outputs1, outputs2])
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=config.lr, amsgrad=True, clipvalue=1.0), loss=['categorical_crossentropy', 'mse'], metrics=['accuracy'])

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
        loadpath = './modelsave/' + '1epochkorean_cnn.h5'
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
