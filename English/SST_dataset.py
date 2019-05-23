import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class SSTDataset():
    def __init__(self, data_path_train: str, max_sequence_length: int):

        with open(data_path_train, 'rt', encoding='utf-8') as f1:
            raw_data = f1.readlines()

        self.reviews = np.array([x.strip().split(' ||| ')[0] for x in raw_data])
        self.labels = np.array([np.float32(float(y.strip().split(' ||| ')[1])) for y in raw_data])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.reviews)
        sequences = tokenizer.texts_to_sequences(self.reviews)
        self.word_index = tokenizer.word_index

        self.reviews = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
        self.sentiment = to_categorical(np.asarray(self.labels))

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx], self.sentiment[idx]

    def shuffle(self):
        order = np.random.permutation(len(self.reviews))
        self.reviews = self.reviews[order]
        self.sentiment = self.sentiment[order]


if __name__ == "__main__":

    def _batch_loader(iterable, n=1):
        length = len(iterable)
        for n_idx in range(0, length, n):
            yield iterable[n_idx:min(n_idx + n, length)]

    print('pre-loading data')

    dset = SSTDataset('../Data/train', 30)
    word_idx_ = dset.word_index

    print("Found %s unique tokens. " % len(word_idx_))

    print(len(dset))

    for epoch in range(2):
        dset.shuffle()
        for i, (data, sentiments) in enumerate(_batch_loader(dset, 5)):
            if i % 100 == 0:
                print('review:  ', data)
                print('sentiment    ', sentiments)