from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras.layers.recurrent import SimpleRNN
import numpy as np

with open('train.csv', "r") as f:
    data = f.read().splitlines()

X_tr = []
Y_tr = []
for record in data:
    if record[0] == "I":
        continue
    X_tr.append(record.split(',')[1])
    Y_tr.append(record.split(',')[2])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(Y_tr)
encoded = tokenizer.texts_to_sequences(Y_tr)[1]

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
#encoded_words = [one_hot(d, vocab_size) for d in Y_tr]


