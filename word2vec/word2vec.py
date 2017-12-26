import numpy as np
from .utils import *
#from utils import *
import sys

class Word2Vec:
    """
    Python implementation of Word2Vec.

    # Arguments
        method : `str`
            choose method for word2vec (options: 'cbow', 'skipgram')
            [default: 'cbow']
        window_size: `integer`
            size of window [default: 1]
        n_hidden: `integer`
            size of hidden layer [default: 2]
        n_epochs: `integer`
            number of epochs [default: 1]
        learning_rate: `float` [default: 0.1]
        corpus: `str`
            corpus text
    """
    def __init__(self, method='cbow', window_size=1, n_hidden=2, n_epochs=1, corpus='', learning_rate=0.1):
        self.window = window_size
        self.N = n_hidden
        self.n_epochs = n_epochs
        self.corpus = [corpus]
        self.eta = learning_rate
        if method == 'cbow':
            self.method = self.cbow
        elif method == 'skipgram':
            self.method = self.skipgram
        else:
            raise ValueError("Method not recognized. Aborting.")

    def cbow(self, context, label, W1, W2, loss):
        h = np.mean([np.dot(W1.T, x) for x in context], axis=0)
        u = np.dot(W2.T, h)
        y_pred = softmax(u)

        e = -label + y_pred
        dW2 = np.outer(h, e)
        dW1 = np.mean([np.outer(x, np.dot(W2, e)) for x in context], axis=0)  # np.outer(x, np.dot(w2, e))

        new_W1 = W1 - self.eta * dW1
        new_W2 = W2 - self.eta * dW2

        loss += -float(u[label == 1]) + np.log(np.sum(np.exp(u)))

        return new_W1, new_W2, loss

    def skipgram(self, context, x, W1, W2, loss):
        h = np.dot(W1.T, x)
        u = np.dot(W2.T, h)
        y_pred = softmax(u)

        e = np.array([-label + y_pred.T for label in context])
        dW2 = np.outer(h, np.sum(e, axis=0))
        dW1 = np.outer(x, np.dot(W2, np.sum(e, axis=0).T))

        new_W1 = W1 - self.eta * dW1
        new_W2 = W2 - self.eta * dW2

        loss += -np.sum([u[label == 1] for label in context]) + len(context) * np.log(np.sum(np.exp(u)))

        return new_W1, new_W2, loss

    def run(self):
        if len(self.corpus)==0:
            raise ValueError('You need to specify a corpus of text.')

        corpus_tokenized, V = tokenize(self.corpus)
        W1, W2 = initialize(V, self.N)

        loss_vs_epoch = []
        for e in range(self.n_epochs):
            loss = 0.
            for context, label in corpus2io(corpus_tokenized, V, self.window):
                W1, W2, loss = self.method(context, label, W1, W2, loss)
            loss_vs_epoch.append(loss)

        return W1, W2, loss_vs_epoch


# cbow = Word2Vec(method="cbow", corpus="I like playing football with my friends", n_epochs=1, learning_rate=0.8)
# W1, W2, loss_vs_epoch = cbow.run()
# print(W1, W2, loss_vs_epoch)
