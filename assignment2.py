import numpy as np
from scipy.sparse import csr_matrix, save_npz, vstack
import time
import os
import re
from string import punctuation
from collections import Counter
from math import log10
import warnings
warnings.filterwarnings("ignore")


start = time.time()
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
    table = str.maketrans("'", ' ', punctuation)
    text = text.lower().translate(table)

    return text.split()


def prepare_text_data(data_type='train', test=False):
    print("[INFO] data {} loading".format(data_type))
    train_texts = load_data('{}.texts'.format(data_type))
    if not test:
        train_labels = load_data('{}.labels'.format(data_type))
    
    # stop words
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
             'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
             'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
             'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
             'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'I', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
             'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
             'wasn', 'weren', 'won', 'wouldn', 'br', 'movie', 'also', 'film']
    
    X = [set(tokenize(text)) for text in train_texts]
    vocab_list = [word for text in X for word in text]
    counter = Counter(vocab_list)
    vocab = set(vocab_list).difference(set(stopwords))
    print("vocab len:", len(set(vocab)))
    rare_words = [x for (x, y) in counter.items() if y == 1]
    vocab = vocab.difference(set(rare_words))
    vocab_dict = {word: i for i, word in enumerate(vocab)}
    print("vocab len after removing rare words:", len(set(vocab)))
    
    #coocurrence_matrix = csr_matrix((len(X), len(vocab)), dtype=np.uint8)
    coocurrence_vectors = [] 
    for i, x in enumerate(X):
        counter = Counter(x)
        col = np.array([vocab_dict[word] for word in counter.keys() if word in vocab])
        row = np.array([0 for _ in range(len(col))])
        data = np.array([freq for word, freq in counter.items() if word in vocab])
        
        coocurrence_vectors.append(csr_matrix((data, (row, col)), shape=(1, len(vocab))))
    
    coocurrence_matrix = vstack(coocurrence_vectors)
    print(coocurrence_matrix.shape)
    if not test:
        Y = train_labels
    else:
        Y = None
    
    return vocab, X, Y

train_vocab, trainX, trainY = prepare_text_data(data_type='train')
#dev_vocab, devX, devY = prepare_text_data(data_type='dev')
#test_vocab, testX, _ = prepare_text_data(data_type='test', test=True)

print("Program time;", time.time() - start)


def sigmoid(x):
    return 1/(1 - np.exp(-x))




