import numpy as np


def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

print("[INFO] glove vector downloading")
with open('../../datasets/glove/glove.6B.50d.txt', 'r') as f:
    data = f.read().splitlines()

glove_dict = {}
for vector in data:
    tokens = vector.split(' ')
    vector =  np.array([float(item) for item in tokens[1:]])
    glove_dict[tokens[0]] = vector / np.linalg.norm(vector)

a = 0
for word, vector in glove_dict.items():
    for word_, vector_ in glove_dict.items():
        if word != word_:
            a = np.dot(vector, vector_)



















