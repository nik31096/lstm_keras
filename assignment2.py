import numpy as np
from scipy.sparse import csr_matrix, save_npz, vstack, hstack
import time
import os
from sys import exit
from string import punctuation
from collections import Counter
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


def prepare_text_data(data_type='train'):
    print("[INFO] data {} loading".format(data_type))
    if data_type == 'train_dev':
        train = load_data('train.texts')
        train_len = len(train)
        dev = load_data('dev.texts')
        dev_len = len(dev)
        texts = train + dev
        print("Len of texts is:", len(texts))
        train_labels = load_data('train.labels')
        dev_labels = load_data('dev.labels')
    else:
        texts = load_data('{}.texts'.format(data_type))

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
    trainY = np.array([0 if item == 'neg' else 1 for item in train_labels])
    devY = np.array([0 if item == 'neg' else 1 for item in dev_labels])

    if data_type == 'train_dev':
        return vocab, hstack([csr_matrix(np.ones((train_len, 1))), coocurrence_matrix[:train_len]], format='csr'), \
               hstack([csr_matrix(np.ones((dev_len, 1))), coocurrence_matrix[train_len:]], format='csr'), trainY, devY

    return vocab, coocurrence_matrix, Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def weights_init(len_vocab):
    weights = np.zeros((1, len_vocab))

    return weights


def loss_calculate(weights, X, Y):
    W_X = sigmoid(X.dot(weights.T)).reshape(-1)
    loss = -1 / N * np.sum(Y.dot(np.log2(W_X)) + (1 - Y).dot(np.log2(1 - W_X))) + alpha * np.sum(weights[1:] ** 2)

    return loss, W_X


def loss_gradient(weights, X, Y):
    nabla_L = 2 * alpha * weights + 1 / N * X.T.dot(sigmoid(X.dot(weights.T).reshape(-1)) - Y)

    return nabla_L.reshape(-1)


def sgd(weights, X, Y):
    random_M_indices = np.random.randint(0, N, 20)
    W_X = sigmoid(X[random_M_indices].dot(weights.T)).reshape(-1)
    nabla_L = 1 / N * X[random_M_indices].T.dot(W_X - Y[random_M_indices]) + 2 * alpha * weights

    return nabla_L


def fit(weights, trainX, trainY, ep2show, eps=1e-7):
    count = 0
    while True:
        # iteration_start = time.time()
        gradient = sgd(weights, trainX, trainY)
        # print("shapes: weights {} and gradient {}".format(weights.shape, gradient.shape))
        if count < 20000:
            lr = 0.5
        else:
            lr = 5e-2
        weights -= lr * gradient
        if np.linalg.norm(gradient) < eps:
            break
        if count % ep2show == 0:
            loss, W_X = loss_calculate(weights, trainX, trainY)
            print("iteration {}, loss: {}, gradient: {}".format(count, loss, np.linalg.norm(gradient)))
        count += 1
        # print("Iteration {} ends with time {}".format(count, time.time() - iteration_start))

    return weights, count


def get_accuracy_on(testX, testY, weights):
    W_X = sigmoid(testX.dot(weights.T)).reshape(-1)
    preds = [1 if item >= 0.5 else 0 for item in W_X]
    count = 0
    for y_pred, y_true in zip(preds, testY):
        if y_pred == y_true:
            count += 1

    return count / len(testY)


train_dev_vocab, trainX, devX, trainY, devY = prepare_text_data(data_type='train_dev')
# dev_vocab, devX, devY = prepare_text_data(data_type='dev')
# test_vocab, testX, _ = prepare_text_data(data_type='test', test=True)
print("Data preparation takes {} seconds".format(round(time.time() - start, 4)))

alpha = 1e-7
N = trainX.shape[0]
V = trainX.shape[1]
print(N, V)
weights = weights_init(V)

trained_weights, train_iterations = fit(weights, trainX, trainY, ep2show=100000)
accuracy_train = get_accuracy_on(trainX, trainY, trained_weights)
accuracy_dev = get_accuracy_on(devX, devY, trained_weights)
print("After {} iterations train accuracy is {}, dev accuracy is {}".format(train_iterations,
                                                                            accuracy_train,
                                                                            accuracy_dev))

print("Program time:", time.time() - start)
