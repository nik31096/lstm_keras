from sys import exit
import os
import re
import random
from collections import Counter
from math import exp, log
from sklearn.metrics import classification_report, accuracy_score
import time

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


train_texts, train_labels = load_data('train.texts'), load_data('train.labels')
dev_texts, dev_labels = load_data('dev.texts'), load_data('dev.labels')
print(dev_labels[:5])
test_texts = load_data('test.texts')

class NaiveBayes:
    def __init__(self, alpha, clf_type='multinomial'):
        self.alpha = alpha
        self.clf_type = clf_type
        
    def fit(self, trainX, trainY):
        Y_distrib = Counter(trainY)
        self.train = {c: Counter([word for text in [x for x, y in zip(trainX, trainY) if y == c] for word in text.lower().split()])
                            for c in Y_distrib.keys()}
        self.amountOfWords = [len(x.values()) for x in self.train.values()]
        self.D_c = [item for item in Y_distrib.values()]
        self.D = len(trainY)
        self.V = len(set([word for text in [x.keys() for x in self.train.values()] for word in text]))
        print("Суммарное количество слов", self.amountOfWords)
        print("Количество документов, принадлежащий соответствующем классам", self.D_c)
        print("Общее количество документов", self.D)
        print("Размер словаря", self.V)
        
    def predict(self, testX):
        preds = []
        for text in testX:
            w = []
            for c, counter in self.train.items():
                tmp = []
                for word in text.split(' '):
                    tmp.append(counter[word])
                w.append(tmp)
            C = [ (log(self.D_c[c]/self.D) + sum([log((self.alpha + w_i)/(self.alpha*self.V + self.amountOfWords[c])) for w_i in w[c]]) ) for c in self.train.keys()]
            preds.append(C.index(max(C)))

        return preds

    @staticmethod
    def softmax(classes_coefficients):
        return [ exp(c_i)/sum([exp(c) for c in classes_coefficients]) for c_i in classes_coefficients ]


def tokenize(text):                                                                                                     
    text = re.sub(r'[^\w\s]', '', text)                                                                                 
    return text


#trainX = ["предоставляю услуги бухгалтера", "спешите купить обои", "надо купить молока"]
#trainY = [0, 0, 1]
#testX = ["предоставляю возможность купить обои по скидке"]
#clf = NaiveBayes(1)
#clf.train(trainX, trainY)
#preds = clf.predict(testX)

print("[INFO] data tokenorization and preparation")
trainX = [tokenize(r) for r in train_texts]
trainY = [0 if item == 'neg' else 1 for item in train_labels]
devX = [tokenize(r) for r in dev_texts]
devY = [0 if item == 'neg' else 1 for item in dev_labels]
testX = [tokenize(r) for r in test_texts]

clf = NaiveBayes(0.1)
print("[INFO] fitting classifier")
clf.fit(trainX, trainY)
print("[INFO] evaluating classifier")
preds = clf.predict(devX)

print(accuracy_score(devY, preds))
end = time.time()

print("Evaluation time: {}".format(end - start))

