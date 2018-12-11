import numpy as np
from sys import exit


class MaxValues:
    def __init__(self, n, l, axis=0):
        self.axis = axis
        self.n = n
        if axis == 0:
            self.l = sorted(l, reverse=True)[:n]
        else:
            self.l = sorted(l, key=lambda x: x[axis], reverse=True)[:n]
    
    def append(self, value):
        if len(self.l) < self.n:
            self.l.insert(0, value)
            if self.axis == 0:
                self.l.sort(reverse=True)
            else:
                self.l.sort(key=lambda x: x[self.axis], reverse=True)
        else:
            if self.axis == 0 and value < self.l[-1]:
                return False
            elif self.axis != 0 and value[self.axis] < self.l[-1][self.axis]:
                return False

            self.l.pop()
            self.l.insert(0, value)
            if self.axis == 0:
                self.l.sort(reverse=True)
            else:
                self.l.sort(key=lambda x: x[self.axis], reverse=True)
        return True

    def __repr__(self):
        return str(self.l)


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

def get_closest(n):
    closest = MaxValues(n, [], axis=1)
    set_of_words = set()
    for word, vector in glove_dict.items():
        for word_, vector_ in glove_dict.items():
            if (word + ':' + word_) or (word_ + ':' + word) in set_of_words:
                continue
            if word != word_:
                set_of_words.add(word + ' : ' + word_)
                a = np.dot(vector, vector_)
                closest.append((word + ' : ' + word_, a))
        
    return [item[0] for item in closest.l]

list_of_words = ['break-of-gauge', 'installations', 'lashkar', 'munt', 'oceanarium',
                 'arraign', 'rand', 'modula-2', 'pit-houses', 'unintimidating']

get_closest(26)

