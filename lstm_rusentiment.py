from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from pandas import read_csv, concat
from sklearn.metrics import classification_report
import numpy as np

# useful links on tutorials: 
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
# https://keras.io/preprocessing/text/#tokenizer


def data_preprocessing(df):
    labels = list(df['label'].values)
    num_labels = dict((v, k) for k, v in dict(enumerate(np.unique(labels))).items())
    labels = to_categorical([num_labels[label] for label in labels])
    _filter = '0123456789\n!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\''
    sentences = [text_to_word_sequence(text=phrase, filters=_filter) for phrase in list(df["text"].values)]
    phrases = []
    for sentence in sentences:
        phrases.append(' '.join(sentence))
    vocabulary = []
    for sentence in sentences:
        for word in sentence:
            vocabulary.append(word)
    vocabulary = sorted(list(set(vocabulary)))

    return phrases, vocabulary, labels  


def encoding(vocabulary, phrases, max_length):
    vocab_size = len(vocabulary)
    encoded_docs = [one_hot(d, vocab_size) for d in phrases]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return vocab_size, padded_docs


data_path = '/home/nik-96/Documents/git/rusentiment/Dataset/'

# train data
df1 = read_csv(data_path + 'rusentiment_preselected_posts.csv')
df2 = read_csv(data_path + 'rusentiment_random_posts.csv')
df = concat([df1, df2])

train_phrases, train_vocabulary, train_labels = data_preprocessing(df)

# test data
df_test = read_csv(data_path + 'rusentiment_test.csv')
test_phrases, test_vocabulary, test_labels = data_preprocessing(df_test)

#t = Tokenizer(num_words=len(vocabulary))
#t.fit_on_texts(sentences)
#print(t.word_counts)
#print(t.document_count)
#print(t.word_index)
#print(t.word_docs)

max_length = max([len(phrase) for phrase in train_phrases])
train_vocab_size, train_padded_docs = encoding(train_vocabulary, train_phrases, max_length)
test_vocab_size, test_padded_docs = encoding(test_vocabulary, test_phrases, max_length)


def simple_model():
    model = Sequential()
    model.add(Embedding(train_vocab_size, 128, input_length=max_length))    
    model.add(Flatten())
    model.add(Dense(5, activation='sigmoid'))

    return model


def lstm_model(hidden_size):
    model = Sequential()
    model.add(Embedding(train_vocab_size, 128, input_length=max_length))
    model.add(LSTM(hidden_size))
    model.add(Dense(5, activation='softmax'))

    return model


model = lstm_model(500)
# compile the model                          
print("[INFO] training network...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])    
# summarize the model                                                           
print(model.summary())    
model.fit(train_padded_docs, train_labels, epochs=3)    
# evaluate the model                              
#loss, accuracy = model.evaluate(padded_docs, labels, verbose=0) 
#print('Accuracy: %f' % (accuracy*100))
print("[INFO] evaluating network...")
preds = model.predict(test_padded_docs)
print(classification_report(test_labels.argmax(axis=1), preds.argmax(axis=1)))

