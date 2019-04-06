from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
import numpy as np
import os


# 이 파일의 결과는 text가 들어오면 embedding 결과를 리턴.

MAX_NB_WORDS = 20
MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 100
GLOVE_DIR = './Glove'

texts = ['A warm , funny , engaging film .', "It 's a lovely film with lovely performances by Buy and Accorsi ."]
labels_index = {'A warm , funny , engaging film .': 0,
                "It 's a lovely film with lovely performances by Buy and Accorsi .": 1}
labels = [4, 3]

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens. " % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# 4 -> [0., 0., 0., 0., 1.]
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


embedding_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    f_ = f.readlines()
    for line in f_:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

print('Found %s word vectors. ' % len(embedding_index))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Embedding이 잘 되는지 확인.
embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False)
print()