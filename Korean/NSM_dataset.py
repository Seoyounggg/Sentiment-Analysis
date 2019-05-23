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

        self.reviews = preprocess(self.reviews, max_length)

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


def preprocess(data: list, max_length: int):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    vectorized_data = [decompose_str_as_one_hot(datum, warning=False) for datum in data]
    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding[idx, :length] = np.array(seq)
    return zero_padding


if __name__ == '__main__':
    a = NSMDataset('../Data/ratings_test.txt', 20)
    print()
