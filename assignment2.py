import numpy as np
from scipy.sparse import csr_matrix, save_npz, vstack
import time
import os
import re
from string import punctuation
from collections import Counter
from math import log2
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
    if data_type == 'train,dev':
        train = load_data('train.texts')

    train_texts = load_data('{}.texts'.format(data_type))
    if not test:
        train_labels = load_data('{}.labels'.format(data_type))
    
    # stop words
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'I', 'should',
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
                 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
                 'br', 'movie', 'also', 'film']
    
    X = [tokenize(text) for text in train_texts]
    vocab_list = [word for text in X for word in text]
    counter = Counter(vocab_list)
    vocab = set(vocab_list).difference(set(stopwords))
    print("vocab len:", len(set(vocab)))
    rare_words = [x for (x, y) in counter.items() if y == 1]
    vocab = vocab.difference(set(rare_words))
    vocab_dict = {word: i for i, word in enumerate(vocab)}
    print("vocab len after removing rare words:", len(set(vocab)))
    
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
        Y = np.array([0 if item == 'neg' else 1 for item in train_labels])
    else:
        Y = None
    
    return vocab, coocurrence_matrix, Y


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def weights_init(len_vocab):
    weights = np.zeros((1, len_vocab))

    return weights


def loss_calculate(weights, X, Y):
    W_X = np.array([sigmoid(X[i].dot(weights.transpose())[0]) for i in range(N)]).reshape(-1)
    W_2 = np.sum([weight*weight for weight in weights])
    loss = -1/N*np.sum([y*log2(w_x) + (1-y)*log2(1-w_x) for w_x, y in zip(W_X, Y)]) + alpha*W_2

    return loss, W_X


def loss_gradient(weights, X, Y, W_X):
    #nabla_L = np.array([2*alpha*weight + 1/N*np.sum([(w_x-y)*X[i, j] for w_x, y, i in zip(W_X, Y, range(N))])
    #                    for weight, j in zip(weights, range(V))])
    nabla_L = 2*alpha*weights + 1/N*X.transpose().dot(W_X-Y)

    return nabla_L.reshape(-1)


def sgd(weights, X, Y, W_X):
    M = 20
    random_M_indices = [np.random.randint(X.shape[0]) for _ in range(M)]
    #nabla_L = np.array([2*alpha*weight + 1/N*np.sum([(W_X[i]-Y[i])*X[i, j] for i in random_M_indices])
    #                    for weight, j in zip(weights, range(V))])
    nabla_L = 1/N*X[random_M_indices].transpose().dot(W_X[random_M_indices]-Y[random_M_indices]) + 2*alpha*weights
    
    return nabla_L.reshape(-1)


def fit(weights, trainX, trainY):
    count = 0
    while True:
        #loss, W_X = loss_calculate(weights, trainX, trainY)
        if count % 10000 == 0:
            loss, W_X = loss_calculate(weights, trainX, trainY)
            print("iteration {}, loss: {}".format(count, loss))
        gradient = loss_gradient(weights, trainX, trainY, W_X)
        # gradient = sgd(weights, trainX, trainY, W_X)
        weights_new = weights - lr*gradient
        weights = weights_new
        count += 1

    return weights_new


# def predict(weights, testX):
#


train_vocab, trainX, trainY = prepare_text_data(data_type='train')
# dev_vocab, devX, devY = prepare_text_data(data_type='dev')
# test_vocab, testX, _ = prepare_text_data(data_type='test', test=True)

print("Data preparation takes {} seconds".format(round(time.time() - start, 2)))

lr = 5e-2
alpha = 1e-7
N = trainX.shape[0]
V = trainX.shape[1]

weights = weights_init(V)

trained_weights = fit(weights, trainX, trainY)
print(np.linalg.norm(trained_weights))

print("Program time:", time.time() - start)
