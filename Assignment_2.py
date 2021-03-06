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
    table = str.maketrans("'", ' ', punctuation + '1234567890')
    text = text.lower().translate(table)

    return text.split()


def prepare_text_data(data_type='train', train_params=None):
    # train_params={'vocab': vocab, "vocab_dict": vocab_dict}
    print("[INFO] data {} loading".format(data_type))
    texts = load_data('{}.texts'.format(data_type))
    labels = load_data('{}.labels'.format(data_type))
    # stop words
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'although'
                 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'I', 'should',
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
                 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
                 'br', 'movie', 'also', 'film', 'whose', 'others', 'outer', 'thus', 'could', "though"]

    X = [tokenize(text) for text in texts]
    vocab_list = [word for text in X for word in text]
    counter = Counter(vocab_list)
    vocab = set(vocab_list).difference(set(stopwords))
    print("vocab len:", len(vocab))
    rare_words = [x for (x, y) in counter.items() if y < 1]
    vocab = vocab.difference(set(rare_words))
    print(tuple(vocab)[-20:])

    print("vocab len after removing rare words:", len(vocab))
    if train_params:
        train_vocab = train_params["vocab"]
        train_vocab_dict = train_params["vocab_dict"]
        print("testing data preparation")
        dev_train_intersection = vocab.intersection(train_vocab)
        print("train dev intersection len", len(dev_train_intersection))
        weights_indices = np.array([train_vocab_dict[word] for word in dev_train_intersection]) + 1
        print("len of indices list for weights: {}".format(len(weights_indices)))
        vocab = dev_train_intersection
        #vocab_dict = {word: train_vocab_dict[word] for word in dev_train_intersection}

    vocab_dict = {word: i for i, word in enumerate(vocab)}
    coocurrence_vectors = []
    for x in X:
        counter = Counter(x)
        col = np.array([vocab_dict[word] for word in counter.keys() if word in vocab])
        row = np.array([0 for _ in range(len(col))])
        data = np.array([freq for word, freq in counter.items() if word in vocab])
       
        coocurrence_vectors.append(csr_matrix((data, (row, col)), shape=(1, len(vocab))))

    coocurrence_matrix = hstack([csr_matrix(np.ones((len(texts), 1))), vstack(coocurrence_vectors)], format='csr')
    Y = np.array([0 if item == 'neg' else 1 for item in labels])
    print("shape of coocurrence_matrix", coocurrence_matrix.shape)
    return vocab, (weights_indices if train_params else vocab_dict), coocurrence_matrix, Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def weights_init(len_vocab):
    weights = np.zeros((len_vocab))

    return weights


def loss_calculate(weights, X, Y, alpha):
    W_X = sigmoid(X.dot(weights.T))
    loss = -1 / N * np.sum(Y.dot(np.log2(W_X)) + (1 - Y).dot(np.log2(1 - W_X))) + alpha * np.sum(weights[1:] ** 2)

    return loss, W_X


def sgd(weights, X, Y, alpha, batch_size):
    random_M_indices = np.random.randint(0, N, batch_size)
    M_of_X = X[random_M_indices]
    W_X = sigmoid(M_of_X.dot(weights.T))
    nabla_L = 1 / N * M_of_X.T.dot(W_X - Y[random_M_indices]) + 2 * alpha * weights

    return nabla_L


def fit(weights, trainX, trainY, alpha, ep2show, eps=1e-6):
    count = 0
    while count < 2000001:
        # iteration_start = time.time()
        gradient = sgd(weights, trainX, trainY, alpha, batch_size=16)
        # print("shapes: weights {} and gradient {}".format(weights.shape, gradient.shape))
        if count < 25000:
            lr = 5e-1
        elif count >= 20000 and count < 1000000:
            lr = 25e-2
        else:
            lr = 8e-3
        weights -= lr * gradient
        if np.linalg.norm(gradient) < eps:
            break
        if count % ep2show == 0:
            iteration_start = time.time()
            loss, W_X = loss_calculate(weights, trainX, trainY, alpha)
            print("iteration {}, loss: {}, gradient: {}".format(count, loss, np.linalg.norm(gradient)))
            print("{} iterations ends with time {}".format(count, time.time() - iteration_start))
        count += 1
        # print("Iteration {} ends with time {}".format(count, time.time() - iteration_start))

    return weights, count


def predict(testX, testY, params=None):
    # params={'weights': weights, 'indices': weigths_indices}
    weights = params['weights']
    if len(params) > 1:
        weights_indices = params['indices']
        W_X = sigmoid(testX.dot(weights.T[[0] + list(weights_indices)])).reshape(-1)
    else:
        W_X = sigmoid(testX.dot(weights.T)).reshape(-1)
    preds = [1 if item >= 0.5 else 0 for item in W_X]
    count = 0
    for y_pred, y_true in zip(preds, testY):
        if y_pred == y_true:
            count += 1

    return count / len(testY)


train_vocab, train_vocab_dict, trainX, trainY = prepare_text_data(data_type='train', train_params=None)
dev_vocab, weights_indices, devX, devY = prepare_text_data(data_type='dev',
                                                           train_params={"vocab": train_vocab,
                                                                         "vocab_dict": train_vocab_dict})
print(len(dev_vocab), devX.shape, len(weights_indices))
# test_vocab, testX, _ = prepare_text_data(data_type='test', test=True)
print("Data preparation takes {} seconds".format(round(time.time() - start, 4)))
alpha = 5e-06
N = trainX.shape[0]
V = trainX.shape[1]
print(N, V)
weights = weights_init(V)
print(weights.shape)
print("[INFO] alpha={}".format(alpha))
one_alpha_time_start = time.time()
trained_weights, train_iterations = fit(weights, trainX, trainY, alpha, ep2show=100000)

accuracy_train = predict(trainX, trainY, params={'weights': trained_weights})
accuracy_dev = predict(devX, devY, params={'weights': trained_weights, 'indices': weights_indices})
print("After {} iterations train accuracy is {}, dev accuracy is {}".format(train_iterations,
                                                                            accuracy_train,
                                                                            accuracy_dev))
print("[INFO] for alpha={} evaluation time is {}".format(alpha, time.time() - one_alpha_time_start))
print("Program time:", time.time() - start)

