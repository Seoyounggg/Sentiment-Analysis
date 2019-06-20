from SST_dataset import SSTDataset
from GloVe_embedding import get_glove, glove_matrix
import argparse
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Activation
from keras import optimizers


def _batch_loader(iterable, n=1):
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # Data path
    args.add_argument('--train_path', type=str, default='../Data/train')
    args.add_argument('--dev_path', type=str, default='../Data/dev')
    args.add_argument('--test_path', type=str, default='../Data/test')

    # options
    args.add_argument('--max_sequence_length', type=int, default=30)
    args.add_argument('--embedding_dim', type=int, default=300)
    args.add_argument('--glove_dir', type=str, default='./Glove/glove.6B.300d.txt')
    args.add_argument('--lstm_size', type=int, default=10)

    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=60)
    args.add_argument('--lr', type=float, default=0.005)
    args.add_argument('--savemodel', type=bool, default=True)
    args.add_argument('--savename', type=str, default='BiLSTM.h5')
    args.add_argument('--mode', type=str, default='train')

    config = args.parse_args()

    # Loading data
    train_data = SSTDataset(config.train_path, config.max_sequence_length)
    dev_data = SSTDataset(config.dev_path, config.max_sequence_length)

    print('Total train dataset:   ', len(train_data))
    print('Total dev dataset:     ', len(dev_data))

    embedding_idx = get_glove(glove_path=config.glove_dir)
    embedding_matrix = glove_matrix(word_idx=train_data.word_index, embedding_idx=embedding_idx,
                                    embedding_dim=config.embedding_dim)

    # model build
    model = Sequential()
    model.add(Embedding(len(train_data.word_index) + 1, config.embedding_dim, weights=[embedding_matrix],
                        input_length=config.max_sequence_length, trainable=False))
    model.add(Bidirectional(LSTM(config.lstm_size, return_sequence=True), merge_mode='concat'))
    model.add(Bidirectional(LSTM(config.lstm_size, return_sequence=False)))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=config.lr), metrics=['accuracy'])
    print(model.summary())

    # train
    if config.mode == 'train':

        train_one_batch = len(train_data)//config.batch
        dev_one_batch = len(dev_data)//config.batch
        best_acc = 0.0

        for epoch in range(config.epochs):

            avg_train_loss = 0.0
            avg_train_acc = 0.0
            train_data.shuffle()

            for i, (data, sentiments) in enumerate(_batch_loader(train_data, config.batch)):
                train_loss, train_acc = model.train_on_batch(data, sentiments)

                if i % 10 == 0:
                    print('Batch : ', i, '/', train_one_batch,
                          ', loss in minibatch: ', float(train_loss),
                          ', acc in minibatch: ', float(train_acc),
                          'current best: ', best_acc)

                avg_train_loss += float(train_loss)
                avg_train_acc += float(train_acc)

                if i % 100 == 0:
                    avg_dev_acc = 0.0
                    dev_data.shuffle()

                    for j, (data_, sentiments_) in enumerate(_batch_loader(dev_data, config.batch)):
                        _, dev_acc = model.test_on_batch(data_, sentiments_)
                        avg_dev_acc += float(dev_acc)

                    cur_acc = avg_dev_acc/dev_one_batch

                    print('Epoch : ', epoch, 'Batch : ', i, '/', train_one_batch, 'Validation ACC : ', cur_acc)

                    if cur_acc >= best_acc and config.savemodel == True:
                        best_acc = cur_acc

                        print('###################  Best Acc Found  #############')
                        model.save('./modelsave/{}epoch'.format(epoch) + config.savename)
                        print('Save new model  {}epoch{}'.format(epoch, config.savename))

            print('\nEpoch: ', epoch,' Train_loss: ', float(avg_train_loss/train_one_batch),
                  ' train_acc:', float(avg_train_acc/train_one_batch),'\n')

        print('best dev acc: ', best_acc)

    else:
        loadpath = './modelsave/' + '1epochBiLSTM.h5'
        model.load_weights(loadpath)

        test_data = SSTDataset(config.test_path, config.max_sequence_length)
        test_one_batch = len(test_data) // config.batch
        print('Total test dataset:    ', len(test_data))

        avg_test_acc = 0.0

        for k, (data_2, sentiments_2) in enumerate(_batch_loader(test_data, config.batch)):
            _, test_acc = model.test_on_batch(data_2, sentiments_2)
            avg_test_acc += float(test_acc)

        cur_acc = avg_test_acc / test_one_batch

        print('Test ACC : ', cur_acc)
