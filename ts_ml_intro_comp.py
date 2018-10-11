from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences, skipgrams
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import SimpleRNN, Dense, Flatten, LSTM, Dropout, GRU
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras.layers.recurrent import SimpleRNN
import numpy as np

def data_extracting(filename):
    with open(filename, "r") as f:
        data = f.read().splitlines()

    X = []
    Y = []
    for record in data:
        if record[0] == "I":
            continue
        X.append(record.split(',')[1])
        Y.append(record.split(',')[2])

    return X, Y

X_tr, Y_tr = data_extracting('train.csv')
Y = ' '.join(Y_tr)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([Y])
encoded = tokenizer.texts_to_sequences([Y])[0]
vocab_size = len(tokenizer.word_index) + 1
sequences = list()                                                               
for i in range(1, len(encoded), 2):
    sequence = encoded[i-1:i+1]
    sequences.append(sequence)
# split into X and y elements
sequences = np.array(sequences)
print(sequences.shape)
X, y = sequences[:, 0], sequences[:, 1]
y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 256, input_length=1))
model.add(SimpleRNN(100))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
adam = Adam(0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=50, verbose=1, batch_size=96)

