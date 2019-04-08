from keras.layers import Embedding
import numpy as np
import os


# 이 파일은 text가 들어오면 embedding 결과를 리턴.

def get_glove(glove_path):

    embedding_idx = {}

    with open(os.path.join(glove_path)) as f:
        f_ = f.readlines()
        for line in f_:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_idx[word] = coefs

    print('Found %s word vectors. ' % len(embedding_idx))

    return embedding_idx


def glove_matrix(word_idx, embedding_idx, embedding_dim):

    embedding_matrix = np.zeros((len(word_idx) + 1, embedding_dim))

    for word, i in word_idx.items():
        embedding_vector = embedding_idx.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


if __name__ == '__main__':

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical

    MAX_SEQUENCE_LENGTH = 10
    EMBEDDING_DIM = 100
    GLOVE_DIR = './Glove'

    texts = ['A warm , funny , engaging film .', "It 's a lovely film with lovely performances by Buy and Accorsi ."]
    labels_index = {'A warm , funny , engaging film .': 0,
                    "It 's a lovely film with lovely performances by Buy and Accorsi .": 1}
    labels = [4, 3]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print("Found %s unique tokens. " % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # 4 -> [0., 0., 0., 0., 1.]
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    embedding_index = get_glove(glove_path=GLOVE_DIR)

    embedding_matrix = glove_matrix(word_idx=word_index, embedding_idx=embedding_index, embedding_dim=EMBEDDING_DIM)

    # Embedding이 잘 되는지 확인.
    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH, trainable=False)

    print()