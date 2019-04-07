import os
import numpy as np
from keras.utils import to_categorical


class SSTDataset():
    def __init__(self, data_path: str):

        data = os.path.join(data_path)

        with open(data, 'rt', encoding='utf-8') as f1:
            raw_data = f1.readlines()

            self.reviews = np.array([x.strip().split(' ||| ')[0] for x in raw_data])
            self.labels = np.array([np.float32(float(y.strip().split(' ||| ')[1])) for y in raw_data])

        self.sentiment = to_categorical(np.asarray(self.labels))

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx], self.sentiment[idx]

    def shuffle(self):
        order = np.random.permutation(len(self.reviews))
        self.reviews = self.reviews[order]
        self.labels = self.labels[order]
        self.sentiment = self.sentiment[order]


if __name__ == "__main__":

    def _batch_loader(iterable, n=1):
        length = len(iterable)
        for n_idx in range(0, length, n):
            yield iterable[n_idx:min(n_idx + n, length)]

    print('pre-loading data')

    dset = SSTDataset('./Data/dev')

    print(len(dset))

    for epoch in range(2):
        dset.shuffle()
        for i, (data, labels, sentiments) in enumerate(_batch_loader(dset, 5)):
            if i % 100 == 0:
                print('review:  ', data)
                print('label:   ', labels)
                print('sentiment    ', sentiments)