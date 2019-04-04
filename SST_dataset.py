import os
import numpy as np


class SSTDataset():
    def __init__(self, dataset_path: str):

        data = os.path.join(dataset_path)

        with open(data, 'rt', encoding='utf-8') as f1:
            raw_data = f1.readlines()

            self.reviews = np.array([x.strip().split(' ||| ')[0] for x in raw_data])
            self.labels = np.array([np.float32(float(y.strip().split(' ||| ')[1])) for y in raw_data])

        self.sentiment = []

        for label in self.labels:
            if label == 0:
                self.sentiment.append([1., 0., 0., 0., 0.])
            elif label == 1:
                self.sentiment.append([0., 1., 0., 0., 0.])
            elif label == 2:
                self.sentiment.append([0., 0., 1., 0., 0.])
            elif label == 3:
                self.sentiment.append([0., 0., 0., 1., 0.])
            elif label == 4:
                self.sentiment.append([0., 0., 0., 0., 1.])
            else:
                print('Error')

        self.sentiment = np.array(self.sentiment)

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

    for i, (data, labels, sentiments) in enumerate(_batch_loader(dset, 5)):
        if i % 100 == 0:
            print('review:  ', data)
            print('label:   ', labels)
            print('sentiment    ', sentiments)