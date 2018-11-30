import numpy as np
from scipy.sparse import csr_matrix, save_npz, vstack, hstack
import time
import os
from sys import exit
from string import punctuation
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

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


def prepare_text_data(texts_train, labels_train, texts_test):
    texts = texts_train + texts_test
    train_len = len(texts_train)
    test_len = len(texts_test)
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
    
    X = [tokenize(text) for text in texts]
    vocab_list = [word for text in X for word in text]
    print("vocab len:", len(set(vocab_list)))
    counter = Counter(vocab_list)
    vocab = set(vocab_list).difference(set(stopwords))
    print("vocab len:", len(set(vocab)))
    rare_words = [x for x, y in counter.items() if y <= 3]
    print("rare words len:", len(rare_words))
    vocab = vocab.difference(set(rare_words))
    vocab_dict = {word: i for i, word in enumerate(vocab)}
    print("vocab len after removing rare words:", len(vocab))
    
    coocurrence_vectors = []
    for i, x in enumerate(X):
        counter = Counter(x)
        col = np.array([vocab_dict[word] for word in counter.keys() if word in vocab])
        row = np.array([0 for _ in range(len(col))])
        data = np.array([freq for word, freq in counter.items() if word in vocab])
        coocurrence_vectors.append(csr_matrix((data, (row, col)), shape=(1, len(vocab))))
    
    coocurrence_matrix = vstack(coocurrence_vectors)
    trainY = np.array([0 if item == 'neg' else 1 for item in labels_train])

    return hstack([csr_matrix(np.ones((train_len, 1))), coocurrence_matrix[:train_len]], format='csr'), \
               hstack([csr_matrix(np.ones((test_len, 1))), coocurrence_matrix[train_len:]], format='csr'), trainY


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def weights_init(len_vocab):
    weights = np.zeros((1, len_vocab))

    return weights


def loss_calculate(weights, X, Y):
    N = X.shape[0]
    alpha = 5e-6
    W_X = sigmoid(X.dot(weights.T)).reshape(-1)
    loss = -1/N*np.sum(Y.dot(np.log2(W_X)) + (1 - Y).dot(np.log2(1 - W_X))) + alpha*np.sum(weights[1:]**2)

    return loss, W_X


def sgd(weights, X, Y):
    N = X.shape[0]
    alpha = 5e-6
    random_M_indices = [np.random.randint(X.shape[0]) for _ in range(32)]
    W_X = sigmoid(X[random_M_indices].dot(weights.T)).reshape(-1)
    nabla_L = 1/N*X[random_M_indices].T.dot(W_X - Y[random_M_indices]) + 2*alpha*weights
    
    return nabla_L.reshape(-1)


def fit(weights, trainX, trainY, ep2show, end):
    count = 0
    while count < int(end) + 1:
        #loss, W_X = loss_calculate(weights, trainX, trainY)
        if count % ep2show == 0:
            loss, W_X = loss_calculate(weights, trainX, trainY)
            print("iteration {}, loss: {}".format(count, loss))
        # gradient = loss_gradient(weights, trainX, trainY, W_X)
        gradient = sgd(weights, trainX, trainY)
        if count < 100000:
            lr = 5e-1
        elif count >= 100000 and count < 1000000:
            lr = 2e-1
        else: 
            lr = 1e-1
        weights_new = weights - lr*gradient
        #if count % 10000 == 0:
        #    print("weights norm: ", np.linalg.norm(weights_new - weights))
        #if np.linalg.norm(weights_new - weights) < eps:
        weights = weights_new
        count += 1
    return weights_new, count


def train(trainX, trainY):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """
    return {"X": trainX, "Y": trainY}


def classify(test_x, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: dict with data structure: {'train': value, 'D': value, 'D_c': value, 'M': value...
    :return: list of labels corresponding to the given list of texts
    """
    train_x = params["X"]
    train_y = params["Y"]
    trainX, testX, trainY = prepare_text_data(train_x, train_y, test_x)
    V = trainX.shape[1]
    weights = weights_init(V)
    
    trained_weights, train_iterations = fit(weights, trainX, trainY, ep2show=100000, end=2e6)
    preds = ['pos' if item >= 0.5 else 'neg' for item in sigmoid(testX.dot(trained_weights.transpose())).reshape(-1)]
    return preds

