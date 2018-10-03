from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding
from pandas import read_csv, concat
import numpy as np

# useful links on tutorials: 
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
# https://keras.io/preprocessing/text/#tokenizer

data_path = '/home/nik-96/Documents/git/rusentiment/Dataset/'

df1 = read_csv(data_path + 'rusentiment_preselected_posts.csv')
df2 = read_csv(data_path + 'rusentiment_random_posts.csv')
df = concat([df1, df2])

def data_preprocessing(df):
    labels = list(df['label'].values)
    num_labels = dict((v, k) for k, v in dict(enumerate(np.unique(labels))).items())
    labels = [num_labels[label] for label in labels]
    print(max(labels))
    sentences = [text_to_word_sequence(text=phrase, 
                                       filters='0123456789\n!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\'') 
                 for phrase in list(df["text"].values)]
    phrases = []

    for sentence in sentences:
        phrases.append(' '.join(sentence))

    vocabulary = []
    for sentence in sentences:
        for word in sentence:
            vocabulary.append(word)
    vocabulary = sorted(list(set(vocabulary)))

    return phrases, vocabulary, labels


# train data
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

def encoding(vocabulary, phrases)
    vocab_size = len(vocabulary)
    encoded_docs = [one_hot(d, vocab_size) for d in phrases]
    max_length = max([len(phrase) for phrase in phrases])
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return vocab_size, padded_docs


train_vocab_size, train_padded_docs = encoding(train_vocabulary, train_phrases)
test_vocab_size, test_padded_docs = encoding(test_vocabulary, test_phrases)

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_length))    
model.add(Flatten())                                            
model.add(Dense(1, activation='sigmoid'))    
# compile the model                          
print("[INFO] training network...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])    
# summarize the model                                                           
print(model.summary())    
model.fit(train_padded_docs, train_labels, epochs=50, verbose=0)    
# evaluate the model                              
#loss, accuracy = model.evaluate(padded_docs, labels, verbose=0) 
#print('Accuracy: %f' % (accuracy*100))
print("[INFO] evaluating network...")
preds = model.predict()


