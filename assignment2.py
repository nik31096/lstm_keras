import numpy as np
import scipy as sp
import time
import os
import re
from string import punctuation
from collections import Counter

#start = time.time()
data_path = '../filimdb_evaluation/FILIMDB'


def load_data(filename):
    print("Loading {}".format(filename))
    dataPath = os.path.join(data_path, filename)
    with open(dataPath) as input_data:
        lines = input_data.readlines()
        lines = [l.strip() for l in lines]
    print("Data size: {}".format(len(lines)))
    return lines


def tokenize(text):
    text = re.sub("'", ' ', text)
    for p in list(punctuation + '1234567890'):
        text = text.replace(p, '')
        text = text.replace('don', 'done')

    return text.lower()


print("[INFO] data loading")
train_texts, train_labels = load_data('train.texts'), load_data('train.labels')
dev_texts, dev_labels = load_data('dev.texts'), load_data('dev.labels')
print(dev_labels[:5])
test_texts = load_data('test.texts')

# stop words
stopwords_path = "../filimdb_evaluation/Starter_code/stopwords.txt"
with open(stopwords_path, 'r') as f:
    stopwords = f.read().split('\n')

stopwords = stopwords + ['br', 'movie', 'also']

train_tokens = [tokenize(text) for text in train_texts]
vocab = [word for text in train_tokens for word in text.split() if word not in stopwords]
print("vocab len:", len(set(vocab)))
counter = Counter(vocab)
rare_words = [x for (x, y) in counter.items() if y == 1]
start = time.time()
vocab = set(vocab) - set(rare_words)
end = time.time()
print("removing operation in seconds: ", end - start)
#vocab = [word for word in vocab if word not in rare_words]
print("number of rare words:", len(rare_words))
print("vocab len after removing rare words:", len(set(vocab)))




