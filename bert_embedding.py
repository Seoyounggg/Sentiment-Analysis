import sys
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import tokenization
import tensorflow as tf
import modeling


folder = '../multi_cased_L-12_H-768_A-12'
config_path = folder + '/bert_config.json'
checkpoint_path = folder + '/bert_model.ckpt'
vocab_path = folder + '/vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)

sentence = '사과는 맛있다.'
token_sent = tokenizer.tokenize(sentence)
print(token_sent)

token_input = tokenizer.convert_tokens_to_ids(token_sent) # vocab.txt 안에 있는 각 단어의 숫자로 바뀐다.
print(token_input)

token_input = token_input + [0] * (512 - len(token_input))
print(token_input)

print(len(token_input))
