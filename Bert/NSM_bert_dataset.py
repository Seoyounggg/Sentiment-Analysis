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
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from kor_char_parser import decompose_str_as_one_hot
from keras.utils import to_categorical


class NSMDataset():

    def __init__(self, dataset_path: str, max_length: int):

        with open(dataset_path, 'rt', encoding='utf-8') as f1:
            raw_data = f1.readlines()
            raw_data = raw_data[1:]

        self.reviews = np.array([x.strip().split('\t')[1] for x in raw_data])
        self.labels = np.array([np.float32(float(y.strip().split('\t')[2][0])) for y in raw_data])

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


if __name__ == '__main__':
    a = NSMDataset('./Data/ratings_test.txt', 20)
    print()
