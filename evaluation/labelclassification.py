from __future__ import print_function
from sklearn.preprocessing import normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

import time
import numpy as np
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--graph', default='BlogCatalog', help='graph name.')
parser.add_argument('--method', default='lemane_frpca_class', help='method name.')
args = parser.parse_args()

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


class Classifier(object):
    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ['micro', 'macro']
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)

        return results

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X_sup, X, Y, train_precent, seed):
        state = np.random.get_state()
        training_size = int(train_precent * len(X))
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        if len(X_sup) == 0:
            X_train = [X[shuffle_indices[i]] for i in range(training_size)]
            Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
            X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
            Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        else:
            train_index = set(X_sup)
            for i in range(training_size):
                if len(train_index) < training_size:
                    if shuffle_indices[i] not in X_sup:
                        train_index.add(shuffle_indices[i])
                else:
                    break
            X_train = [X[i] for i in list(train_index)]
            Y_train = [Y[i] for i in list(train_index)]
            X_test = [X[i] for i in list(set(range(len(X))) - train_index)]
            Y_test = [Y[i] for i in list(set(range(len(X))) - train_index)]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)

def load_embeddings(filename):
    vectors = np.loadtxt(filename, delimiter=',')
    return vectors


def read_node_label(filename):
    fin = open(filename, "r")
    X = []
    Y = []
    while True:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(int(vec[0]))
        Y.append(vec[1:])
    fin.close()
    return X, Y

def load_sup_info(filename):
    fin = open(filename, "r")
    X_sup = []
    while True:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X_sup.append(int(vec[0]))
    return X_sup

if __name__ == '__main__':

    labelfile1 = 'label/' + args.graph + '.txt'
    labelfile2 = "label/" + args.graph + '_sup.txt'

    clf_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]
    X1, Y = read_node_label(labelfile1)

    embfile1 = 'embds/' + args.graph + '/' + args.graph + '_' + args.method + '_U.csv'
    embfile2 = 'embds/' + args.graph + '/' + args.graph + '_' + args.method + '_V.csv'
    vectors1 = load_embeddings(embfile1)
    vectors2 = load_embeddings(embfile2)
    vectors1 = vectors1[X1]
    vectors2 = vectors2[X1]
    vectors1 = normalize(vectors1, norm='l2')
    vectors2 = normalize(vectors2, norm='l2')
    vectors = np.concatenate((vectors1, vectors2), axis=1)

    print(vectors.shape)

    X_sup = []
    X_sup_index = load_sup_info(labelfile2)
    for i in X_sup_index:
        X_sup.append(X1.index(i))

    X = list(range(int(vectors.shape[0])))

    for ratio in clf_ratio:
        print('Training classifier using {:.2f}% nodes...'.format(ratio * 100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        results = []
        for i in range(5):
            results.append(float(clf.split_train_evaluate(X_sup, X, Y, ratio, i)['micro']))
        print("Micro F1 score: {:.4f}".format(np.mean(results) * 100))
